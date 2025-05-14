import random
from math import sqrt, log
import tempfile
from sys import argv
import interactive_host


class State:
    def __init__(self, decisions=[], score=0.0, visits=0, true_child=None, false_child=None, parent=None):
        self.decisions:list[bool] = decisions
        self.score:float = score
        self.visits:int = visits
        self.true_child:State|None = true_child
        self.false_child:State|None = false_child
        self.parent:State|None = parent
    
    def __repr__(self) -> str:
        return (f"Decisions: {self.decisions}\n"
                f"Score: {self.score}\n"
            f"Visits: {self.visits}\n")

    def is_leaf(self) -> bool:
        return not self.true_child and not self.false_child

class InlineMonteCarloAdvisor(object):
    def __init__(self, C=sqrt(2)):
        self.root:State = State()
        self.current:State = self.root
        self.C:float = C # exploration factor for UCT

    def advice_false(self, _) -> bool:
        return False
    def advice_true(self, _) -> bool:
        return True

    def advice(self, _) -> bool:
        if self.current.is_leaf():
            return self.get_rollout_decision()
        else:
            next = self.get_next_state(self.current) 
            self.current = next
            return next.decisions[-1]

    def get_next_state(self, state:State) -> State:
        children = (state.true_child, state.false_child)
        
        match children:
            case (None, None):
                if self.get_rollout_decision():
                    false_child = State(state.decisions[:] + [False])
                    state.false_child = false_child
                    return false_child
                else:
                    true_child = State(state.decisions[:] + [True])
                    state.true_child = true_child
                    return true_child
            case (_, None):
                false_child = State(state.decisions[:] + [False])
                state.false_child = false_child
                return false_child
            case (None, _):
                true_child = State(state.decisions[:] + [True])
                state.true_child = true_child
                return true_child
            case (true_child, false_child):
                return true_child if self.uct(true_child) > self.uct(false_child) else false_child

    def uct(self, state:State) -> float:
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
                ['opt',
                '-passes=scc-oz-module-inliner',
                '-interactive-model-runner-echo-reply',
                '-enable-ml-inliner=release',
                f"-inliner-interactive-channel-base={filename}.channel-basename",
                '-o', f2.name,
                f1.name
                ])
            optimized_mod = f2.read()
            return scoring_function(optimized_mod)

    def update_score(self, score:float):
        self.current.score = (self.current.score-score)/self.current.visits

    def run_monte_carlo(self, nr_of_turns:int, input_mod, scoring_function):
        for _ in range(nr_of_turns):
            self.current = self.root
            score = self.get_score(input_mod, scoring_function)
            while self.current:
                self.current.visits+=1
                self.update_score(score)
                self.current = self.current.parent
