import unittest

from advisors.merged.merged_mc_advisor import MergedMonteCarloAdvisor


class TestSubtreeFullyExplored(unittest.TestCase):
    def setup_advisor(self) -> MergedMonteCarloAdvisor:
        advisor = MergedMonteCarloAdvisor("test")
        advisor.loop_unroll_advisor.MAX_UNROLL_FACTOR = 3
        return advisor

    def test_root_initially_not_explored(self):
        advisor = self.setup_advisor()
        # on construction, nothing is marked
        self.assertFalse(advisor.root.subtree_is_fully_explored)

    def test_leaf_initially_not_explored(self):
        advisor = self.setup_advisor()
        leaf = advisor.root.add_child(True)
        self.assertFalse(leaf.subtree_is_fully_explored)

    def test_set_leaf_marks_only_leaf(self):
        advisor = self.setup_advisor()
        parent = advisor.root.add_child(True)
        leaf = parent.add_child(True)

        advisor.set_state_as_fully_explored(leaf)

        # only the leaf itself flips
        self.assertTrue(leaf.subtree_is_fully_explored)
        self.assertFalse(parent.subtree_is_fully_explored)
        self.assertFalse(advisor.root.subtree_is_fully_explored)

    def test_parent_marks_when_all_bool_children(self):
        advisor = self.setup_advisor()
        parent = advisor.root.add_child(True)

        # a boolean branch has exactly two children
        leaf_a = parent.add_child(True)
        leaf_b = parent.add_child(False)

        # mark one child → parent still unmarked
        advisor.set_state_as_fully_explored(leaf_a)
        self.assertFalse(parent.subtree_is_fully_explored)

        # mark the second child → parent now flips
        advisor.set_state_as_fully_explored(leaf_b)
        self.assertTrue(leaf_a.subtree_is_fully_explored)
        self.assertTrue(leaf_b.subtree_is_fully_explored)
        self.assertTrue(parent.subtree_is_fully_explored)
        # but root has only one child so far → still False
        self.assertFalse(advisor.root.subtree_is_fully_explored)

    def test_root_marks_when_all_bool_children(self):
        advisor = self.setup_advisor()

        # give the root two boolean branches
        branch_true = advisor.root.add_child(True)
        branch_false = advisor.root.add_child(False)

        # mark only one branch → root stays unmarked
        advisor.set_state_as_fully_explored(branch_true)
        self.assertFalse(advisor.root.subtree_is_fully_explored)

        # mark the other → root flips
        advisor.set_state_as_fully_explored(branch_false)
        self.assertTrue(branch_true.subtree_is_fully_explored)
        self.assertTrue(branch_false.subtree_is_fully_explored)
        self.assertTrue(advisor.root.subtree_is_fully_explored)

    def test_multilevel_propagation(self):
        advisor = self.setup_advisor()

        # 1) build a bool‐branch under root
        level1 = advisor.root.add_child(True)
        # 2) under that, build its two bool‐children
        leaf1 = level1.add_child(True)
        leaf2 = level1.add_child(False)

        # marking both → level1 flips, but root still has only one child so far
        advisor.set_state_as_fully_explored(leaf1)
        advisor.set_state_as_fully_explored(leaf2)
        self.assertTrue(level1.subtree_is_fully_explored)
        self.assertFalse(advisor.root.subtree_is_fully_explored)

        # 3) now give root its second branch
        sibling = advisor.root.add_child(False)
        # mark it
        advisor.set_state_as_fully_explored(sibling)

        # at this point both root‐children are flagged → root flips
        self.assertTrue(advisor.root.subtree_is_fully_explored)

    def test_mixed(self):
        advisor = self.setup_advisor()

        branch_true = advisor.root.add_child(True)
        branch_false = advisor.root.add_child(False)

        c1 = branch_true.add_child(1)
        c3 = branch_true.add_child(3)

        c1_true = c1.add_child(True)
        c1_false = c1.add_child(False)

        advisor.set_state_as_fully_explored(c3)

        self.assertFalse(branch_true.subtree_is_fully_explored)

        advisor.set_state_as_fully_explored(c1_true)
        advisor.set_state_as_fully_explored(c1_false)

        self.assertFalse(branch_true.subtree_is_fully_explored)

        c2 = branch_true.add_child(2)
        advisor.set_state_as_fully_explored(c2)
        self.assertTrue(branch_true.subtree_is_fully_explored)
        self.assertFalse(advisor.root.subtree_is_fully_explored)

        advisor.set_state_as_fully_explored(branch_false)
        self.assertTrue(advisor.root.subtree_is_fully_explored)


if __name__ == "__main__":
    unittest.main()
