from dataclasses import replace
import unittest

from lars.train.data import Record
from sweep.distillation_ablation import retarget


def record(source, target):
    return Record(source, source, "choice", "state", "Pick", ["a", "b"], target)


class DistillationAblationTests(unittest.TestCase):
    def test_lower_mixture_preserves_original_soft_gold(self):
        gold = record("arc", [.8, .2])
        # Existing mix50 is halfway between soft gold and teacher [.6, .4].
        soft = replace(gold, target=[.7, .3])
        result, changed = retarget([gold], [soft], None, .25)
        self.assertAlmostEqual(result[0].target[0], .75)
        self.assertAlmostEqual(result[0].target[1], .25)
        self.assertEqual(changed, {"arc": 1})

    def test_source_arm_preserves_population_metadata_and_other_targets(self):
        gold = [record("arc", [.8, .2]), record("mnli", [.6, .4])]
        soft = [replace(gold[0], target=[.7, .3]), replace(gold[1], target=[.55, .45])]
        result, changed = retarget(gold, soft, ["arc"], .5)
        self.assertEqual(result, [soft[0], gold[1]])
        self.assertEqual(changed, {"arc": 1})
        self.assertEqual(gold[0].target, [.8, .2])

    def test_reordered_or_changed_records_are_rejected(self):
        a, b = record("arc", [.8, .2]), record("mnli", [.6, .4])
        with self.assertRaisesRegex(ValueError, "population/order/metadata"):
            retarget([a, b], [b, a], None, .5)
        with self.assertRaisesRegex(ValueError, "population/order/metadata"):
            retarget([a], [replace(a, question="changed")], None, .5)
        with self.assertRaisesRegex(ValueError, "populations differ"):
            retarget([a], [], None, .5)

    def test_invalid_mixtures_and_targets_are_rejected(self):
        a = record("arc", [.8, .2])
        for mix in [0., -.1, .75, float("nan"), float("inf")]:
            with self.subTest(mix=mix), self.assertRaises(ValueError):
                retarget([a], [a], None, mix)
        with self.assertRaisesRegex(ValueError, "invalid mixed target"):
            retarget([a], [replace(a, target=[1.1, -.1])], None, .5)


if __name__ == "__main__":
    unittest.main()
