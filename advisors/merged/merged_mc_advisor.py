import logging
import tempfile
from math import sqrt
from os import stat
from typing import Any

from typing_extensions import override

from advisors import log_reader
from advisors.inline.inline_mc_advisor import InlineMonteCarloAdvisor
from advisors.loop_unroll.loop_unroll_mc_advisor import LoopUnrollMonteCarloAdvisor
from advisors.mc_advisor import MonteCarloAdvisor, MonteCarloError, State
from advisors.merged.merged_runner import MergedCompilerCommunicator

logger = logging.getLogger(__name__)


class MergedMonteCarloAdvisor(MonteCarloAdvisor[bool | int]):
    def __init__(self, input_name, C=sqrt(2)) -> None:
        super().__init__(input_name)
        self.inline_advisor = InlineMonteCarloAdvisor(input_name)
        self.loop_unroll_advisor = LoopUnrollMonteCarloAdvisor(input_name)
        self.runner = MergedCompilerCommunicator(input_name, False, False)

    def opt_args(self) -> list[str]:
        return [
            "opt",
            "-O3",
            # "-passes=default<O3>,loop-unroll",
            "-interactive-model-runner-echo-reply",
            "-inliner-interactive-include-default",
            # "-debug-only=inline,inline-ml",
            "-enable-ml-inliner=release",
            f"-inliner-interactive-channel-base={self.inline_advisor.filename}.channel-basename",
            f"--mlgo-loop-unroll-interactive-channel-base={self.loop_unroll_advisor.filename}.channel-basename",
            "--mlgo-loop-unroll-advisor-mode=development",
            "-debug-only=loop-unroll-development-advisor,loop-unroll,inline,inline-ml",
        ]

    def get_next_state(self, state: State[bool | int], inline: bool = True) -> State:
        if state.is_leaf():
            choice = self.get_rollout_decision(inline)
            return state.add_child(choice)
        assert (
            type(state.children[0].decisions[-1]) is bool
        ) == inline  # if we have an inlining decision, we expect the children to be inline == bool decision states
        if inline:
            return self.inline_advisor.get_next_state(state)
        else:
            return self.loop_unroll_advisor.get_next_state(state)

    def get_rollout_decision(self, inline: bool = True) -> bool | int:
        if inline:
            return self.inline_advisor.get_rollout_decision()
        else:
            return self.loop_unroll_advisor.get_rollout_decision()

    def get_default_decision(self, tv, heuristic) -> bool | int:
        if heuristic is None:
            return self.inline_advisor.get_default_decision(tv, heuristic)
        else:
            return self.loop_unroll_advisor.get_default_decision(tv, heuristic)

    def check_unroll_success(self, action: bool):
        if (
            not action  # we did not unroll
            and not self.in_rollout  # we dont care about rollouts
            and self.current  # get typechecker to shush
            and self.current.decisions[-1] != 1  # we did want to unroll
        ):

            self.runner.stop_event.set()
            logger.warning("Unsuccessful unrolling")
            raise MonteCarloError("unsuccessful unrolling")

    @override
    def get_initial_tree(self, path: str):
        def build_initial_path(
            tv: list[log_reader.TensorValue] = [], heuristic=None
        ) -> Any:
            assert self.current
            default_decision = self.get_default_decision(tv, heuristic)
            child = self.current.add_child(default_decision)
            child.score = 1.0
            child.speedup_sum = 1.0
            child.visits = 1
            self.current = child
            return self.wrap_advice(default_decision, heuristic is None)

        self.root.score = 1.0
        self.root.speedup_sum = 1.0
        self.root.visits = 1

        self.runner.compile_once(
            self.opt_args() + ["-o", path + "mod-post-mc.bc", path + "mod-pre-mc.bc"],
            build_initial_path,
        )
        assert self.current
        self.default_path = self.current.decisions

    @override
    def advice(self, tv, heuristic) -> Any:
        assert self.current
        if self.current.visits == 0:
            self.in_rollout = True
            decision = self.get_rollout_decision(heuristic is None)
        else:
            next = self.get_next_state(self.current, heuristic is None)
            self.current = next
            decision = next.decisions[-1]
        return self.wrap_advice(decision, heuristic is None)

    @override
    def wrap_advice(
        self, advice: bool | int, inline: bool = True
    ) -> bool | list[float]:
        if inline:
            return advice
        else:
            return self.loop_unroll_advisor.wrap_advice(advice)

    @override
    def get_score(self, path: str, scoring_function):
        self.runner.compile_once(
            self.opt_args() + ["-o", path + "mod-post-mc.bc", path + "mod-pre-mc.bc"],
            self.advice,
            on_action=self.check_unroll_success,
        )
        return scoring_function()
