import logging
from math import sqrt
import random

import tempfile
from typing import final
from typing_extensions import override
import loop_unroll_runner
from mc_advisor import MonteCarloAdvisor, State, MonteCarloError

logger = logging.getLogger(__name__)


@final
class LoopUnrollMonteCarloAdvisor(MonteCarloAdvisor[int]):
    def __init__(self, C: float = sqrt(2)) -> None:
        super().__init__(C)
        self.runner: loop_unroll_runner.LoopUnrollCompilerCommunicator = (
            loop_unroll_runner.LoopUnrollCompilerCommunicator(False, True)
        )

        # These need to be kept in sync with the ones in UnrollModelFeatureMaps.h
        self.MAX_UNROLL_FACTOR = 5
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

    @override
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

    def check_unroll_success(self, action: bool):
        if (
            not action  # we did not unroll
            and not self.in_rollout  # we dont care about rollouts
            and self.current  # get typechecker to shush
            and self.current.decisions[-1] != 1  # we did want to unroll
        ):

            logger.warning("Unsuccessful unrolling")
            raise MonteCarloError("unsuccessful unrolling")

    @override
    def get_score(self, input_mod: bytes, scoring_function):
        filename = type(self).__name__
        with tempfile.NamedTemporaryFile(
            suffix=".ll"
        ) as f1, tempfile.NamedTemporaryFile(suffix=".bc") as f2:
            f1.write(input_mod)
            f1.flush()

            self.runner.compile_once(
                f"{filename}.channel-basename",
                self.advice,
                self.opt_args() + ["-o", f2.name, f1.name],
                on_action=self.check_unroll_success,
            )
            optimized_mod = f2.read()
            return scoring_function(optimized_mod)
