import logging
import tempfile
from abc import ABC, abstractmethod
from math import log, sqrt
from typing import Any, Generic, Optional, TypeVar

from . import log_reader

logger = logging.getLogger(__name__)

D = TypeVar("D", int, bool)


class MonteCarloError(Exception):
    pass


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

    def __repr__(self) -> str:
        return (
            f"State(decisions={self.decisions}, "
            f"score={self.score:.7f},"
            f"visits={self.visits})"
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
    def __init__(self, C: float = sqrt(2)) -> None:
        self.runner: Any
        self.C = C
        self.root = State[D]()
        self.current = self.root
        self.max_node = self.root
        self.in_rollout: bool = False
        self.default_path: list[D]
        self.max_speedup_after_n_iterations: list[float] = [1.0]
        self.filename = type(self).__name__

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

    def advice(self, tv, heuristic) -> Any:
        assert self.current
        if self.current.visits == 0:
            self.in_rollout = True
            decision = self.get_rollout_decision()
        else:
            next = self.get_next_state(self.current)
            self.current = next
            decision = next.decisions[-1]
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
        return scoring_function()

    def update_score(self):
        assert self.current
        samesies = self.current is self.max_node
        if samesies:
            print(f"Max is same as current: {samesies}")
            print(f"Current max: {self.max_node.score}")
        self.current.score = (
            self.current.speedup_sum / self.current.visits
        )  # average speedup
        if self.current.is_leaf():
            self.max_node = max(self.current, self.max_node)
            self.max_speedup_after_n_iterations.append(self.max_node.score)
            print(f"After max: {self.max_node.score}")

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

    def run_monte_carlo(self, nr_of_turns: int, path: str, scoring_function):
        self.get_initial_tree(path)
        logger.info(self)
        logger.info(f"Current Max {self.max_node.score}")
        for i in range(nr_of_turns):
            logger.info(f"Monte Carlo iteration {i}")
            try:
                self.current = self.root
                self.in_rollout = False
                score = self.get_score(path, scoring_function)
                while self.current:
                    self.current.speedup_sum += score
                    self.current.visits += 1
                    self.update_score()
                    self.current = self.current.parent
            except MonteCarloError:
                assert self.current
                self.current.score = -999
                self.current.speedup_sum = -999
                self.current.visits = 1
                self.max_speedup_after_n_iterations.append(
                    max(self.current.score, self.max_node.score)
                )
            except KeyboardInterrupt as k:
                logger.error(k)
                break
            logger.debug(self)
        logger.info(self)
        logger.info(f"Highest scoring: {self.get_max_state()}")
