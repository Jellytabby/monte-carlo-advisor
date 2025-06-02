from __future__ import annotations

import logging
import random
from math import sqrt
from typing import final

from ..mc_advisor import MonteCarloAdvisor, State
from . import inline_runner

logger = logging.getLogger(__name__)


@final
class InlineMonteCarloAdvisor(MonteCarloAdvisor[bool]):
    def __init__(self, C: float = sqrt(2)) -> None:
        super().__init__(C)
        self.runner = inline_runner.InlineCompilerCommunicator()
        self.filename = self.runner.channel_base

    def opt_args(self) -> list[str]:
        return [
            "opt",
            # "-passes=inline",
            "-O3",
            # "-passes=default<O3>,scc-oz-module-inliner",
            "-inliner-interactive-include-default",
            "-interactive-model-runner-echo-reply",
            "-debug-only=inline,inline-ml",
            "-enable-ml-inliner=release",
            f"-inliner-interactive-channel-base={self.filename}.channel-basename",
        ]

    def get_rollout_decision(self) -> bool:
        choice = random.random()
        return True if choice >= 0.5 else False

    def get_default_decision(self, tv, heuristic) -> bool:
        return bool(tv[-1][0])

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
