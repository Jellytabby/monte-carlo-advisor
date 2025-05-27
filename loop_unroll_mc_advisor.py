import logging
from math import sqrt
import random

from typing import final
import loop_unroll_runner
from mc_advisor import MonteCarloAdvisor, State

logger = logging.getLogger(__name__)


@final
class LoopUnrollMonteCarloAdvisor(MonteCarloAdvisor[int]):
    def __init__(self, C: float = sqrt(2)) -> None:
        super().__init__(C)
        self.runner = loop_unroll_runner.LoopUnrollCompilerCommunicator(False, True)

        # These need to be kept in sync with the ones in UnrollModelFeatureMaps.h
        self.MAX_UNROLL_FACTOR = 10
        self.UNROLL_FACTOR_OFFSET = 2
        # self.ADVICE_TENSOR_LEN = 1 + self.MAX_UNROLL_FACTOR - self.UNROLL_FACTOR_OFFSET
        self.ADVICE_TENSOR_LEN = 1 + 32 - self.UNROLL_FACTOR_OFFSET

    def opt_args(self) -> list[str]:
        filename = type(self).__name__
        return [
            "opt",
            "-O3",
            f"--mlgo-loop-unroll-interactive-channel-base={filename}.channel-basename",
            "--mlgo-loop-unroll-advisor-mode=development",
            "--interactive-model-runner-echo-reply",
            "-debug-only=loop-unroll-development-advisor,loop-unroll",
        ]

    def make_response_for_factor(self, factor: int):
        l = [0.5 for _ in range(self.ADVICE_TENSOR_LEN)]
        if factor == 0 or factor == 1:
            return l
        assert factor <= self.MAX_UNROLL_FACTOR
        assert factor >= self.UNROLL_FACTOR_OFFSET
        assert factor - self.UNROLL_FACTOR_OFFSET < self.ADVICE_TENSOR_LEN
        l[factor - self.UNROLL_FACTOR_OFFSET] = 2.0
        return l

    def wrap_advice(self, advice: int) -> list[float]:
        return self.make_response_for_factor(advice)

    def get_rollout_decision(self) -> int:
        return random.randint(1, self.MAX_UNROLL_FACTOR)

    def get_default_decision(self, tv, heuristic: int) -> int:
        return (
            1 if heuristic == 0 else heuristic
        )  # compiler returns 0 when no unrolling

    def get_next_state(self, state: State[int]) -> State[int]:
        if state.is_leaf():
            choice = self.get_rollout_decision()
            return state.add_child(choice)
        if len(state.children) == self.MAX_UNROLL_FACTOR:
            return max(state.children, key=self.uct)
        else:
            visited_unroll_factors = set([c.decisions[-1] for c in state.children])
            remaining_unroll_factors = (
                set(range(1, self.MAX_UNROLL_FACTOR + 1)) - visited_unroll_factors
            )
            return state.add_child(random.choice(list(remaining_unroll_factors)))
