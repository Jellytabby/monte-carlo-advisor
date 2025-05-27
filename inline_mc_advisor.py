from __future__ import annotations
import logging
import random
from typing import final
from mc_advisor import State, MonteCarloAdvisor

logger = logging.getLogger(__name__)


@final
class InlineMonteCarloAdvisor(MonteCarloAdvisor[bool]):
    def opt_args(self) -> list[str]:
        filename = type(self).__name__
        return [
            "opt",
            # "-passes=inline",
            "-O3",
            # "-passes=default<O3>,scc-oz-module-inliner",
            "-inliner-interactive-include-default",
            "-interactive-model-runner-echo-reply",
            # "-debug-only=inline",
            "-enable-ml-inliner=release",
            f"-inliner-interactive-channel-base={filename}.channel-basename",
        ]

    def get_rollout_decision(self) -> bool:
        choice = random.random()
        return True if choice >= 0.5 else False

    def get_next_state(self, state: State[bool]) -> State[bool]:
        if state.is_leaf():
            choice = self.get_rollout_decision()
            return state.add_child(choice)
        elif len(state.children) == 2:
            return max(state.children, key=self.uct)
        elif state.children[0].decisions[-1] == False:
            return state.add_child(True)
        elif state.children[0].decisions[-1] == True:
            return state.add_child(False)
        else:
            assert False

    def extract_default_decision_from_tensor(self, tv) -> bool:
        assert self.current
        default_inilining_decisison = bool(tv[-1][0])
        child = self.current.add_child(default_inilining_decisison)
        child.score = 1.0
        child.speedup_sum = 1.0
        child.visits = 1
        self.current = child
        return default_inilining_decisison
