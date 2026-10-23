from collections import Counter
from dataclasses import replace
import unittest

from lars.train.data import Record
from sweep.data_correction import clean_candidates, content_key, hard_class, refresh_source


def record(rid, label, state=None, source="source", target=None):
    return Record(rid, source, "choice", state or rid, "Pick", ["a", "b", "c"],
                  target or [float(i == label) for i in range(3)])


class DataCorrectionTests(unittest.TestCase):
    def test_normalized_whole_state_overlap_is_blocked(self):
        original = record("old", 0, state={"text": " Hello\n WORLD "})
        candidate = record("new", 1, state={"text": "hello world"})
        selected, rejected = clean_candidates([(candidate, 1)], {content_key(original)}, lambda _: True)
        self.assertEqual(selected, [])
        self.assertEqual(rejected["state_overlap"], 1)

    def test_conflicts_ambiguity_and_hard_soft_disagreement_are_excluded(self):
        a, b = record("a", 0, "same"), record("b", 1, "same")
        ambiguous = record("ambiguous", 0, target=[.5, .3, .2])
        disagree = record("disagree", 1)
        good = record("good", 2, target=[.05, .05, .9])
        selected, rejected = clean_candidates([(a, 0), (b, 1), (ambiguous, 0), (disagree, 0), (good, 2)], set(), lambda _: True)
        self.assertEqual(selected, [good])
        self.assertEqual(rejected["conflicting_hard_labels"], 2)
        self.assertEqual(rejected["ambiguous_or_hard_soft_disagreement"], 2)

    def test_replacements_preserve_size_source_and_original_labels(self):
        original = [record(str(i), 0) for i in range(8)] + [record("minor", 1)]
        candidates = [record("new-" + str(i), 1, target=[.1, .9, 0.]) for i in range(5)]
        output, swaps = refresh_source(original, candidates, {"b": 4}, {"b": 5}, 2)
        self.assertEqual(len(swaps), 2)
        self.assertEqual(len(output), len(original))
        self.assertEqual(Counter(r.source for r in output), Counter(r.source for r in original))
        self.assertEqual(Counter(hard_class(r) for r in output), {"a": 6, "b": 3})
        self.assertEqual(original[0].target, [1., 0., 0.])
        self.assertTrue(all(r in original or r in candidates for r in output))
        self.assertEqual(len({s["inserted"] for s in swaps}), len(swaps))

    def test_low_support_classes_do_not_drive_replacement(self):
        original = [record(str(i), 0) for i in range(5)]
        output, swaps = refresh_source(original, [record("new", 1)], {"b": 1}, {"b": 1}, 3)
        self.assertEqual(output, original)
        self.assertEqual(swaps, [])

    def test_length_rejection_and_determinism(self):
        candidates = [(record("b", 1), 1), (record("a", 0), 0)]
        self.assertEqual(clean_candidates(candidates, set(), lambda _: False)[0], [])
        self.assertEqual(clean_candidates(candidates, set(), lambda _: True)[0],
                         clean_candidates(list(reversed(candidates)), set(), lambda _: True)[0])


if __name__ == "__main__":
    unittest.main()
