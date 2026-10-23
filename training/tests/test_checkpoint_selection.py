import unittest

from sweep.checkpoint_selection import selection_metrics


class CheckpointSelectionTests(unittest.TestCase):
    def test_seen_and_unseen_groups_receive_equal_weight(self):
        # Uneven source and population sizes must not swamp the unseen group.
        rows = {str(i): {"source": source, "kind": "choice", "p": p,
                        "target": [.8, .2], "hard_index": 1}
                for i, (source, p) in enumerate(
                    [("seen-a", [.1, .9])] * 9 + [("seen-b", [.1, .9]), ("unseen", [.9, .1])])}
        result = selection_metrics(rows, {"seen-a", "seen-b"})
        self.assertEqual(result["seen_macro_accuracy"], 1.)
        self.assertEqual(result["unseen_macro_accuracy"], 0.)
        self.assertEqual(result["selection_score"], .5)
        self.assertAlmostEqual(result["macro_accuracy"], 2 / 3)

    def test_missing_unseen_group_is_rejected(self):
        row = {"source": "seen", "kind": "choice", "p": [.9, .1], "target": [1., 0.]}
        with self.assertRaisesRegex(ValueError, "both seen and unseen"):
            selection_metrics({"one": row}, {"seen"})


if __name__ == "__main__":
    unittest.main()
