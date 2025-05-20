from __future__ import annotations
from functools import total_ordering
import logging
import random
from math import sqrt, log
import tempfile
import interactive_host

logger = logging.getLogger(__name__)

@total_ordering
class InlineState:
    def __init__(self, decisions=[], score=0.0, speedup_sum=0.0, visits=0, true_child=None, false_child=None, parent=None):
        self.decisions:list[bool] = decisions
        self.score:float = score
        self.speedup_sum : float = speedup_sum
        self.visits:int = visits
        self.true_child:InlineState|None = true_child
        self.false_child:InlineState|None = false_child
        self.parent:InlineState|None = parent

    def __repr__(self) -> str:
        return (f"State(decisions={self.decisions!r}, "
                f"score={self.score:.7f}," 
                f"visits={self.visits},")

    def repr_subtree(self):
        """
        Print the subtree rooted at `root` in an ASCII-tree layout.
        """
        lines: list[str] = ['\n']

        def _walk(node, prefix: str, label: str|None, is_last: bool):
            # pick branch symbols
            connector = "└── " if is_last else "├── "
            label_str = f"[{label}] " if label else ""
            lines.append(prefix + connector + label_str + repr(node))

            # prepare prefix for children
            new_prefix = prefix + ("      " if is_last else "│     ")

            # gather existing children in order
            children = []
            if node.true_child:
                children.append(("True",  node.true_child))
            if node.false_child:
                children.append(("False", node.false_child))

            # recurse
            for idx, (lbl, child) in enumerate(children):
                _walk(child, new_prefix, lbl, idx == len(children) - 1)

        # print the root node itself (no connector or label)
        lines.append(repr(self))

        # then its immediate children
        root_children = []
        if self.true_child:
            root_children.append(("True",  self.true_child))
        if self.false_child:
            root_children.append(("False", self.false_child))

        for idx, (lbl, child) in enumerate(root_children):
            _walk(child, "", lbl, idx == len(root_children) - 1)
            
        return "\n".join(lines)


    def __eq__(self, other) -> bool:
        if not isinstance(other, InlineState):
            return NotImplemented
        return self.score == other.score

    def __lt__(self, other) -> bool:
        if not isinstance(other, InlineState):
            return NotImplemented
        return self.score < other.score

    def add_child(self, choice:bool):
        child = InlineState(self.decisions[:] + [choice])
        if choice:
            self.true_child = child
        else:
            self.false_child = child
        child.parent = self
        return child

    def is_leaf(self) -> bool:
        return not self.true_child and not self.false_child


class InlineMonteCarloAdvisor(object):
    def __init__(self, C=sqrt(2)):
        self.root:InlineState = InlineState()
        self.current:InlineState|None = self.root
        self.C:float = C # exploration factor for UCT
        self.best_state:InlineState

    def __repr__(self):
        return self.root.repr_subtree()

    def advice(self, _) -> bool:
        assert self.current
        if self.current.visits == 0:
            return self.get_rollout_decision()
        else:
            next = self.get_next_state(self.current) 
            self.current = next
            return next.decisions[-1]

    def get_next_state(self, state:InlineState) -> InlineState:
        children = (state.true_child, state.false_child)
        
        match children:
            case (None, None):
                if self.get_rollout_decision():
                    return state.add_child(False)
                else:
                    return state.add_child(True)
            case (_, None):
                return state.add_child(False)
            case (None, _):
                return state.add_child(True)
            case (true_child, false_child):
                return true_child if self.uct(true_child) > self.uct(false_child) else false_child

    def uct(self, state:InlineState) -> float:
        assert state.parent
        return state.score + self.C * sqrt(log(state.parent.visits)/state.visits)

    def get_rollout_decision(self) -> bool:
        choice = random.random()
        return True if choice >= 0.5 else False

    def get_score(self, input_mod:bytes, scoring_function):
        filename = "mod_temp"
        with tempfile.NamedTemporaryFile(suffix=".ll") as f1, \
            tempfile.NamedTemporaryFile(suffix=".bc") as f2:
            f1.write(input_mod)
            f1.flush()
    
            interactive_host.run_interactive(
                f"{filename}.channel-basename", 
                self.advice,
                # lambda _ : False,
                ['opt',
                "-passes=default<O3>,scc-oz-module-inliner",
                # '-passes=scc-oz-module-inliner',
                '-interactive-model-runner-echo-reply',
                '-debug-only=inline',
                '-enable-ml-inliner=release',
                f"-inliner-interactive-channel-base={filename}.channel-basename",
                '-o', f2.name,
                f1.name
                ])
            optimized_mod = f2.read()
            return scoring_function(optimized_mod)

    def update_score(self):
        assert self.current
        self.current.score = self.current.speedup_sum / self.current.visits # average speedup

    def get_max_leaf_state(self) -> InlineState:
        def get_max_leaf_state_helper(current:InlineState | None, max_state:InlineState) -> InlineState:
            if current is None:
                return max_state
            if current.is_leaf():
                return max(current, max_state)
            return max(get_max_leaf_state_helper(current.true_child, max_state), get_max_leaf_state_helper(current.false_child, max_state))
        return get_max_leaf_state_helper(self.root, self.root)

    def run_monte_carlo(self, nr_of_turns:int, input_mod, scoring_function):
        for _ in range(nr_of_turns):
            self.current = self.root
            score = self.get_score(input_mod, scoring_function)
            while self.current:
                self.current.speedup_sum += score
                self.current.visits+=1
                self.update_score()
                self.current = self.current.parent
            logger.debug(self)
        logger.info(self)
        logger.info(f"Highest scoring: {self.get_max_leaf_state()}")

