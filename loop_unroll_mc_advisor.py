import json
import logging
from math import sqrt
import random

import tempfile
from typing import final
from typing_extensions import override
import interactive_host
from mc_advisor import MonteCarloAdvisor, State
import unrolling_runner

logger = logging.getLogger(__name__)


@final
class LoopUnrollMonteCarloAdvisor(MonteCarloAdvisor[int]):
    def __init__(self, C: float = sqrt(2)) -> None:
        super().__init__(C)

        # These need to be kept in sync with the ones in UnrollModelFeatureMaps.h
        self.MAX_UNROLL_FACTOR = 10
        self.UNROLL_FACTOR_OFFSET = 2
        # self.ADVICE_TENSOR_LEN = 1 + self.MAX_UNROLL_FACTOR - self.UNROLL_FACTOR_OFFSET
        self.ADVICE_TENSOR_LEN = 1 + 32 - self.UNROLL_FACTOR_OFFSET

    def make_response_for_factor(self, factor: int):
        l = [0.5 for _ in range(self.ADVICE_TENSOR_LEN)]
        if factor == 0 or factor == 1:
            return l
        assert factor <= self.MAX_UNROLL_FACTOR
        assert factor >= self.UNROLL_FACTOR_OFFSET
        assert factor - self.UNROLL_FACTOR_OFFSET < self.ADVICE_TENSOR_LEN
        l[factor - self.UNROLL_FACTOR_OFFSET] = 2.0
        return l

    def wrap_advice(self, tv, fv) -> list[float]:
        return self.make_response_for_factor(self.advice(tv))

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

    def get_rollout_decision(self) -> int:
        return random.randint(1, self.MAX_UNROLL_FACTOR)
        # return 5

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

    def extract_default_decision_from_tensor(self, tv) -> int:
        return 5

    def extract_default_decision_heuristic(self, _, heuristic_unrolling_factor: int):
        assert self.current
        child = self.current.add_child(heuristic_unrolling_factor)
        child.score = 1.0
        child.speedup_sum = 1.0
        child.visits = 1
        self.current = child
        return self.make_response_for_factor(heuristic_unrolling_factor)

    def get_initial_tree(self, input_mod: bytes):
        self.root.score = 1.0
        self.root.speedup_sum = 1.0
        self.root.visits = 1

        filename = type(self).__name__
        with tempfile.NamedTemporaryFile(
            suffix=".ll"
        ) as f1, tempfile.NamedTemporaryFile(suffix=".bc") as f2:
            f1.write(input_mod)
            f1.flush()

            runner = unrolling_runner.UnrollCompilerHost(False, True)

            runner.compile_once(
                f"{filename}.channel-basename",
                self.extract_default_decision_heuristic,
                self.opt_args() + ["-o", f2.name, f1.name],
                lambda index, features: (),
                lambda index, heuristic: (),
                lambda index, action: (),
                lambda index: None,
            )

    @override
    def get_score(self, input_mod: bytes, scoring_function):
        filename = type(self).__name__
        with tempfile.NamedTemporaryFile(
            suffix=".ll"
        ) as f1, tempfile.NamedTemporaryFile(suffix=".bc") as f2:
            f1.write(input_mod)
            f1.flush()

            runner = unrolling_runner.UnrollCompilerHost(False, True)

            runner.compile_once(
                f"{filename}.channel-basename",
                self.wrap_advice,
                self.opt_args() + ["-o", f2.name, f1.name],
                lambda index, features: (),
                lambda index, heuristic: (),
                lambda index, action: (),
                lambda index: None,
            )
            optimized_mod = f2.read()
            return scoring_function(optimized_mod)
