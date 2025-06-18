import logging
import random
import tempfile
from math import sqrt
from typing import final

from typing_extensions import override

from advisors.loop_unroll import loop_unroll_runner
from advisors.mc_advisor import MonteCarloAdvisor, State
from utils import MonteCarloError

logger = logging.getLogger(__name__)


@final
class LoopUnrollMonteCarloAdvisor(MonteCarloAdvisor[int]):
    def __init__(self, input_name: str, C: float = sqrt(2)) -> None:
        super().__init__(input_name, C)
        self.runner: loop_unroll_runner.LoopUnrollCompilerCommunicator = (
            loop_unroll_runner.LoopUnrollCompilerCommunicator(input_name, True)
        )
        self.filename = self.runner.channel_base

        self.MAX_UNROLL_FACTOR = 2

    def opt_args(self) -> list[str]:
        return [
            "opt",
            # "-O3",
            "-passes=default<O3>,loop-unroll",
            f"--mlgo-loop-unroll-interactive-channel-base={self.runner.channel_base}",
            "--mlgo-loop-unroll-advisor-mode=development",
            "--interactive-model-runner-echo-reply",
            "-debug-only=loop-unroll-development-advisor,loop-unroll",
        ]

    def make_response_for_factor(self, factor: int):
        assert factor >= -1
        return factor

    @override
    def wrap_advice(self, advice: int) -> list[float]:
        return self.make_response_for_factor(advice)

    def get_rollout_decision(self) -> int:
        return random.randint(1, self.MAX_UNROLL_FACTOR)

    def get_default_decision(self, tv, heuristic: int) -> int:
        match heuristic:
            case -1:  # compiler returns -1 when no unrolling
                return 1
            case 0:
                return 1  # ngl, not sure if this is even possible
            case heuristic if heuristic > self.MAX_UNROLL_FACTOR:
                return self.MAX_UNROLL_FACTOR
            case heuristic:
                return heuristic

    def set_state_as_fully_explored(self, state: State[int]):
        state.subtree_is_fully_explored = True
        current = state.parent
        while current:
            if len(current.children) == self.MAX_UNROLL_FACTOR and all(
                c.subtree_is_fully_explored for c in current.children
            ):
                current.subtree_is_fully_explored = True
                current = current.parent
            else:
                return

    def get_next_state(self, state: State[int]) -> State[int]:
        if state.is_leaf():
            choice = self.get_rollout_decision()
            return state.add_child(choice)
        if len(state.children) == self.MAX_UNROLL_FACTOR:
            return max(
                (c for c in state.children if not c.subtree_is_fully_explored),
                key=self.uct,
            )
        else:
            visited_unroll_factors = set([c.decisions[-1] for c in state.children])
            remaining_unroll_factors = (
                set(range(1, self.MAX_UNROLL_FACTOR + 1)) - visited_unroll_factors
            )
            return state.add_child(random.choice(list(remaining_unroll_factors)))

    def check_unroll_success(self, action: bool):
        if action:
            return
        if self.in_rollout and self.current_path[-1] != 1:
            # we are in rollout, so we don't want to block off our mc state, but want the path to accurately reflect what choices we made
            self.current_path[-1] = 1
            return
        if not self.in_rollout and self.current and self.current.decisions[-1] != 1:
            assert self.current.is_leaf()
            # should only happen in leaf states, otherwise would have triggered earlier
            logger.warning("Unsuccessful unrolling")
            raise MonteCarloError("unsuccessful unrolling")

    @override
    def get_score(self, path: str, scoring_function):
        self.runner.compile_once(
            self.opt_args() + ["-o", path + "mod-post-mc.bc", path + "mod-pre-mc.bc"],
            self.advice,
            on_action=self.check_unroll_success,
        )
        return scoring_function()
