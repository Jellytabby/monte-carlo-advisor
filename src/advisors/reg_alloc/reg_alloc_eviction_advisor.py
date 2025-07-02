import random
from math import sqrt
from typing import Optional, final

from advisors import log_reader
from advisors.mc_advisor import MonteCarloAdvisor, State
from advisors.reg_alloc.reg_alloc_eviction_runner import (
    RegAllocEvictionCompilerCommunicator,
)


@final
class RegAllocEvictionMonteCarloAdvisor(MonteCarloAdvisor[int]):
    def __init__(
        self, input_name: str, path: str, timeout: Optional[float], C: float = sqrt(2)
    ):
        super().__init__(input_name, path, timeout, C)
        self.runner = RegAllocEvictionCompilerCommunicator(input_name, True)

    def opt_args(self) -> list[str]:
        return [
            "llc",
            "-O3",
            "-filetype=obj",
            "-regalloc-enable-advisor=release",
            "-interactive-model-runner-echo-reply",
            f"-regalloc-evict-interactive-channel-base={self.runner.channel_base}",
        ] + ["-o", self.path + "mod-post-mc.o", self.path + "mod-post-mc.bc"]

    def get_rollout_decision(
        self, tv: list[log_reader.TensorValue], heuristic: Optional[int]
    ) -> int:
        return random.choice(self.get_possible_registers(tv))

    def get_default_decision(
        self,
        advisor_type: str,
        tv: list[log_reader.TensorValue],
        heuristic: Optional[int],
    ) -> int:
        return self.get_possible_registers(tv)[
            0
        ]  # FIX: this is not actual defualt behavior need to fix

    def get_next_state(
        self,
        state: State[int],
        tv: list[log_reader.TensorValue],
        heuristic: Optional[int],
    ) -> State[int]:
        if state.is_leaf():
            choice = self.get_rollout_decision(tv, heuristic)
            return state.add_child(choice)
        possible_registers = self.get_possible_registers(tv)
        if len(state.children) == len(possible_registers):

            return max(
                (c for c in state.children if not c.subtree_is_fully_explored),
                key=self.uct,
            )
        else:
            visited_registers = set([c.decisions[-1] for c in state.children])
            remaining_registers = set(possible_registers) - visited_registers
            return state.add_child(random.choice(list(remaining_registers)))

    def set_state_as_fully_explored(self, state: State[int]):
        return

    def get_possible_registers(self, tv: list[log_reader.TensorValue]) -> list[int]:
        mask = next(v for v in tv if v.spec().name == "mask")
        registers = []
        for i, value in enumerate(mask):
            if value:
                registers.append(i)
        return registers
