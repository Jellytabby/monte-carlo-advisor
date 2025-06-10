import unittest

from advisors.mc_advisor import MonteCarloAdvisor, State


# A dummy concrete advisor for testing get_max_state
class DummyAdvisor(MonteCarloAdvisor[int]):
    def opt_args(self) -> list[str]:
        return []

    def get_rollout_decision(self) -> int:
        return 0

    def get_next_state(self, state: State[int]) -> State[int]:
        return state

    def get_default_decision(self, tv, heuristic) -> int:
        return 0


class TestGetMaxState(unittest.TestCase):
    def test_empty_tree(self):
        adv = DummyAdvisor()
        # No children: root is the only node
        adv.root.score = 1.5
        adv.root.visits = 1
        max_state = adv.get_max_state()
        self.assertIs(max_state, adv.root)

    def test_single_level_children(self):
        adv = DummyAdvisor()
        # Two children with distinct scores
        c1 = adv.root.add_child(0, score=2.0, speedup_sum=2.0, visits=1)
        c2 = adv.root.add_child(1, score=3.0, speedup_sum=3.0, visits=1)
        max_state = adv.get_max_state()
        self.assertIs(max_state, c2)

    def test_nested_children(self):
        adv = DummyAdvisor()
        # Nested grandchildren scenario
        c1 = adv.root.add_child(0, score=1.0, speedup_sum=1.0, visits=1)
        gc = c1.add_child(0, score=5.0, speedup_sum=5.0, visits=1)
        c2 = adv.root.add_child(1, score=4.0, speedup_sum=4.0, visits=1)
        max_state = adv.get_max_state()
        self.assertIs(max_state, gc)

    def test_root_higher_than_children(self):
        adv = DummyAdvisor()
        # Root has the highest score
        adv.root.score = 10.0
        adv.root.visits = 1
        c1 = adv.root.add_child(0, score=2.0, speedup_sum=2.0, visits=1)
        max_state = adv.get_max_state()
        self.assertIs(max_state, adv.root)

    def test_sophia(self):
        adv = DummyAdvisor()
        adv.root.score = 7

        c8 = adv.root.add_child(0, 8.0)
        c3 = c8.add_child(0, 3.0)

        c1 = adv.root.add_child(0, 1)
        c2 = c1.add_child(0, 2)
        c6 = c1.add_child(0, 6)
        max = adv.get_max_state()
        self.assertIs(max, c8)

    def test_sophia_2(self):
        adv = DummyAdvisor()
        adv.root.score = 2

        c6 = adv.root.add_child(0, 6)
        c15 = c6.add_child(0, 15)
        c4 = c6.add_child(0, 4)

        c10 = adv.root.add_child(0, 10)
        c12 = c10.add_child(0, 12)
        c13 = c10.add_child(0, 13)

        c11 = c13.add_child(0, 11)
        c9 = c11.add_child(0, 9)

        max = adv.get_max_state()
        self.assertIs(max, c15)


if __name__ == "__main__":
    unittest.main()
