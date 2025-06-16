import logging
import subprocess
from abc import ABC, abstractmethod
from math import log, sqrt
from typing import Any, Generic, Optional, TypeVar

import utils
from advisors import log_reader

logger = logging.getLogger(__name__)

D = TypeVar("D", int, bool)


class State(Generic[D]):
    def __init__(
        self,
        decisions=None,
        score=0.0,
        speedup_sum=0.0,
        visits=0,
        parent=None,
        children=None,
    ):
        self.decisions: list[D] = [] if decisions is None else decisions
        self.score: float = score
        self.speedup_sum: float = speedup_sum
        self.visits: int = visits
        self.parent: Optional["State"] = parent
        self.children: list["State"] = [] if children is None else children
        self.subtree_is_fully_explored: bool = False

    def __repr__(self) -> str:
        return (
            f"State(decisions={self.decisions}, "
            f"score={self.score:.7f},"
            f"visits={self.visits})" + (f"*" if self.subtree_is_fully_explored else "")
        )

    def __getitem__(self, index: D):
        return next(c for c in self.children if c.decisions[-1] == index)

    def repr_subtree(self):
        """
        Print the subtree rooted at `root` in an ASCII-tree layout.
        """
        lines: list[str] = ["\n"]

        def _walk(node: "State", prefix: str, label: str | None, is_last: bool):
            # pick branch symbols
            connector = "└── " if is_last else "├── "
            label_str = f"[{label}] " if label else ""
            lines.append(prefix + connector + label_str + repr(node))

            # prepare prefix for children
            new_prefix = prefix + ("      " if is_last else "│     ")

            # gather existing children in order
            labeled_children = []
            for child in node.children:
                labeled_children.append((str(child.decisions[-1]), child))

            # recurse
            for idx, (lbl, child) in enumerate(labeled_children):
                _walk(child, new_prefix, lbl, idx == len(labeled_children) - 1)

        # print the root node itself (no connector or label)
        lines.append(repr(self))

        # then its immediate children
        root_children = []
        for child in self.children:
            root_children.append((str(child.decisions[-1]), child))

        for idx, (lbl, child) in enumerate(root_children):
            _walk(child, "", lbl, idx == len(root_children) - 1)

        return "\n".join(lines)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.score == other.score

    def __lt__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.score < other.score

    def add_child(
        self, choice: D, score: float = 0.0, speedup_sum=0.0, visits: int = 0
    ) -> "State[D]":
        """Create, link, and return a new child state."""
        child = State(
            self.decisions[:] + [choice],
            score=score,
            speedup_sum=speedup_sum,
            visits=visits,
        )
        self.children.append(child)
        self.children.sort(key=lambda c: c.decisions[-1])
        child.parent = self
        return child

    def is_leaf(self) -> bool:
        return self.children == []


class MonteCarloAdvisor(ABC, Generic[D]):
    def __init__(
        self,
        input_name: str,
        C: float = sqrt(2),
    ) -> None:
        self.runner: Any
        self.C = C
        self.root = State[D]()
        self.current = self.root
        self.in_rollout: bool = False
        self.default_path: list[D]
        self.current_path: list[D] = []
        self.all_runs: list[tuple[list[D], float]] = []
        self.invalid_paths: set[tuple[D]] = set()
        self.max_speedup_after_n_iterations: list[float] = [1.0]
        self.filename = input_name

    def __repr__(self):
        return self.root.repr_subtree()

    @abstractmethod
    def opt_args(self) -> list[str]:
        """Return the specific opt flags for this advisor."""
        ...

    @abstractmethod
    def get_rollout_decision(self) -> D: ...

    @abstractmethod
    def get_next_state(self, state: State[D]) -> State: ...

    @abstractmethod
    def get_default_decision(self, tv, heuristic) -> D: ...

    @abstractmethod
    def set_state_as_fully_explored(self, state: State[D]): ...

    def wrap_advice(self, advice: D) -> Any:
        "Wrapper method in case the compiler expects a different data structure than we are storing in our State"
        return advice

    def get_initial_tree(self, path: str):
        def build_initial_path(
            tv: list[log_reader.TensorValue] = [], heuristic=None
        ) -> Any:
            assert self.current
            default_decision = self.get_default_decision(tv, heuristic)
            child = self.current.add_child(default_decision, 1.0, 1.0, 1)
            self.current = child
            return self.wrap_advice(default_decision)

        self.root.score = 1.0
        self.root.speedup_sum = 1.0
        self.root.visits = 1

        self.runner.compile_once(
            self.opt_args() + ["-o", path + "mod-post-mc.bc", path + "mod-pre-mc.bc"],
            build_initial_path,
        )
        assert self.current
        self.default_path = self.current.decisions
        self.all_runs.append((self.default_path[:], 1.0))

    def advice(self, tv, heuristic) -> Any:
        assert self.current
        if self.current.visits == 0:
            self.in_rollout = True
            decision = self.get_rollout_decision()
        else:
            next = self.get_next_state(self.current)
            assert next
            self.current = next
            decision = next.decisions[-1]
        self.current_path.append(decision)
        logger.debug(f"Current path: {self.current_path}")
        return self.wrap_advice(decision)

    def uct(self, state: State) -> float:
        parent = state.parent
        assert parent and state.visits > 0
        return state.score + self.C * sqrt(log(parent.visits) / state.visits)

    def get_score(self, path: str, scoring_function):
        self.runner.compile_once(
            self.opt_args() + ["-o", path + "mod-post-mc.bc", path + "mod-pre-mc.bc"],
            self.advice,
        )
        if self.current_path in self.invalid_paths:
            raise subprocess.TimeoutExpired("no", 1)
        return scoring_function()

    def update_score(self, score: float):
        assert self.current
        self.current.speedup_sum += score
        self.current.visits += 1
        self.current.score = (
            self.current.speedup_sum / self.current.visits
        )  # average speedup

        if len(self.current.decisions) == len(
            self.default_path
        ):  # if we have as many decisions as the default path _in_ the node, then we have reached the bottom of the tree
            self.set_state_as_fully_explored(self.current)

    def get_max_state(self) -> State:
        def get_max_state_helper(current: State | None, max_state: State) -> State:
            if current is None:
                return max_state
            if current.is_leaf():
                return max(current, max_state)

            max_state = max(current, max_state)
            return max(
                map(lambda s: get_max_state_helper(s, max_state), current.children)
            )

        return get_max_state_helper(self.root, self.root)

    def get_max_run(self) -> tuple[list[D], float]:
        return max(self.all_runs, key=lambda x: x[1])

    def mark_state_as_invalid(self, state: State[D]):
        state.score = -999
        state.speedup_sum = -999
        state.visits = 1
        self.set_state_as_fully_explored(state)
        self.all_runs.append((self.current_path[:], -999))
        self.max_speedup_after_n_iterations.append(
            self.max_speedup_after_n_iterations[-1]
        )

    def run_monte_carlo(self, nr_of_turns: int, path: str, scoring_function):
        self.get_initial_tree(path)
        logger.info(self)
        max_score = 1.0
        for i in range(nr_of_turns):
            logger.info(f"Monte Carlo iteration {i}")
            if self.root.subtree_is_fully_explored:
                logger.info("Explored the entire tree!")
                break
            try:
                self.current = self.root
                self.current_path = []
                self.in_rollout = False
                score = self.get_score(path, scoring_function)
                while self.current:
                    self.update_score(score)
                    self.current = self.current.parent
                self.all_runs.append((self.current_path[:], score))
                max_score = max(max_score, score)
                self.max_speedup_after_n_iterations.append(max_score)
            except utils.MonteCarloError: # should happen if we have an invalid loop unroll while not in rollout
                self.mark_state_as_invalid(self.current)
            except (subprocess.TimeoutExpired, TimeoutError): # should happen if opt/llc times out
                assert self.current
                self.invalid_paths.add(tuple(self.current_path[:]))
                if self.current.decisions == self.current_path:
                    self.mark_state_as_invalid(
                        self.current
                    )  # we timed out in a tree node
                logger.warning(
                    f"State: {self.current} with decisions {self.current_path} timed out."
                )
                # TODO: find some way to penalize llc/opt that takes too long, so that we avoid exploring that path again
            except KeyboardInterrupt as k:
                logger.error(f"Received keyboard interrupt {k}")
                break
            # logger.debug(self)
        logger.info(self)
        logger.info(f"Highest scoring decisions: {self.get_max_run()}")
