from typing_extensions import override

from advisors.inline.inline_mc_advisor import InlineMonteCarloAdvisor
from advisors.loop_unroll.loop_unroll_mc_advisor import \
    LoopUnrollMonteCarloAdvisor
from advisors.mc_advisor import MonteCarloAdvisor, State
from advisors.merged.merged_runner import MergedCompilerCommunicator


class MergedMonteCarloAdvisor(MonteCarloAdvisor[bool | int]):
    def __init__(self, C) -> None:
        super().__init__(C)
        self.inline_advisor = InlineMonteCarloAdvisor()
        self.loop_unroll_advisor = LoopUnrollMonteCarloAdvisor()
        self.runner = MergedCompilerCommunicator(False, True)

    def opt_args(self) -> list[str]:
        return [
            "opt",
            "-O3",
            # "-passes=default<O3>,loop-unroll",
            "-interactive-model-runner-echo-reply",
            # "-inliner-interactive-include-default",
            # "-debug-only=inline,inline-ml",
            # "-enable-ml-inliner=release",
            # f"-inliner-interactive-channel-base={filename}.channel-basename",
            f"--mlgo-loop-unroll-interactive-channel-base={self.loop_unroll_advisor.filename}.channel-basename",
            "--mlgo-loop-unroll-advisor-mode=development",
            "-debug-only=loop-unroll-development-advisor,loop-unroll,inline,inline-ml",
        ]

    def get_next_state(self, state: State[bool | int]) -> State:
        # if inline:
        #     return self.inline_advisor.get_next_state(state)
        # else:
        return self.loop_unroll_advisor.get_next_state(state)

    def get_rollout_decision(self) -> bool | int:
        # if inline:
        #     return self.inline_advisor.get_rollout_decision()
        # else:
        return self.loop_unroll_advisor.get_rollout_decision()

    def get_default_decision(self, tv, heuristic) -> bool | int:
        if tv:
            return self.inline_advisor.get_default_decision(tv, heuristic)
        else:
            return self.loop_unroll_advisor.get_default_decision(tv, heuristic)
    
    @override
    def wrap_advice(self, advice: bool|int) -> bool|list[float]:
        # if inline:
        #     return advice
        # else:
            return self.loop_unroll_advisor.wrap_advice(advice)

