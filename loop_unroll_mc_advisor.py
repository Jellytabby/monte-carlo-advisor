import random

from typing import final
from mc_advisor import MonteCarloAdvisor, State


@final
class LoopUnrollMonteCarloAdvisor(MonteCarloAdvisor[int]):
    def __init__(self, C: float, max_unroll_factor: int = 16) -> None:
        super().__init__(C)
        self.max_unroll_factor = max_unroll_factor

    def opt_args(self) -> list[str]:
        filename = type(self).__name__
        return ["opt"]

    def get_rollout_decision(self) -> int:
        return random.randint(0, self.max_unroll_factor)

    def get_next_state(self, state: State[int]) -> State[int]:
        if state.is_leaf():
            choice = self.get_rollout_decision()
            return state.add_child(choice)
        if (
            len(state.children) == self.max_unroll_factor + 1
        ):  # unroll factors + do not unroll (=> factor = 0)
            return max(state.children, key=self.uct)
        else:
            visited_unroll_factors = set([c.decisions[-1] for c in state.children])
            remaining_unroll_factors = (
                set(range(self.max_unroll_factor + 1)) - visited_unroll_factors
            )
            return state.add_child(random.choice(list(remaining_unroll_factors)))

    def extract_default_decision_from_tensor(self, tv) -> int:
        return 0
