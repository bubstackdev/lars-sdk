import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from lars.train.accuracy import distill, record_key, report
from lars.train.data import Record


def record(rid="a", kind="score", state=None):
    return Record(rid, "source", kind, state or {"text": rid}, "How urgent?",
                  ["low", "medium", "high"], [0., 1., 0.])


def prediction(r, p):
    return {"key": record_key(r), "source": r.source, "kind": r.kind, "target": r.target, "p": p}


class AccuracyTests(unittest.TestCase):
    def test_distillation_excludes_shared_states_even_with_different_questions(self):
        a, b = record(), record("b")
        excluded = replace(a, id="other", question="Another question")
        teacher = {record_key(r): prediction(r, [.1, .8, .1]) for r in [a, b]}
        student = {record_key(r): prediction(r, [.6, .3, .1]) for r in [a, b]}
        result, manifest = distill([a, b], teacher, student, [excluded])
        self.assertEqual([r.id for r in result], ["b"])
        np.testing.assert_allclose(result[0].target, [.05, .9, .05])
        self.assertEqual(manifest["excluded_by_state"], 1)
        self.assertEqual(b.target, [0., 1., 0.])

    def test_wrong_teacher_keeps_gold(self):
        r = record()
        cap = {record_key(r): prediction(r, [.8, .1, .1])}
        result, manifest = distill([r], cap, cap, [])
        self.assertEqual(result[0].target, r.target)
        self.assertEqual(manifest["distilled"], 0)

    def test_targeting_keeps_population_and_prioritizes_mistakes(self):
        a, b = record(), record("b")
        teacher = {record_key(r): prediction(r, [.1, .8, .1]) for r in [a, b]}
        student = {record_key(a): prediction(a, [.1, .8, .1]), record_key(b): prediction(b, [.7, .2, .1])}
        result, _ = distill([a, b], teacher, student, [], per_source=1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].target, a.target)
        self.assertNotEqual(result[1].target, b.target)

    def test_report_rejects_mismatched_populations_and_gold(self):
        r = record()
        row = prediction(r, [.1, .8, .1])
        with self.assertRaises(ValueError):
            report({"a": {"a": row}, "b": {}})
        with self.assertRaises(ValueError):
            report({"a": {"a": row}, "b": {"a": {**row, "target": [1., 0., 0.]}}})

    def test_paired_report_and_score_errors(self):
        r = record()
        result = report({"base": {"a": prediction(r, [.8, .1, .1])},
                         "new": {"a": prediction(r, [.1, .8, .1])}})
        self.assertEqual(result["models"]["new"]["paired_accuracy"]["ci95"], [1., 1.])
        self.assertEqual(result["models"]["base"]["all"]["score_mae"], 1.)

    def test_teacher_quality_gate_keeps_other_sources_unchanged(self):
        r = record()
        cap = {record_key(r): prediction(r, [.1, .8, .1])}
        result, manifest = distill([r], cap, cap, [], allowed_sources=set())
        self.assertEqual(result[0].target, r.target)
        self.assertEqual(manifest["distilled"], 0)

    def test_paired_comparison_uses_matching_seed(self):
        r = record()
        result = report({"baseline-s0": {"a": prediction(r, [.8, .1, .1])},
                         "baseline-s1": {"a": prediction(r, [.1, .8, .1])},
                         "ordinal-s1": {"a": prediction(r, [.1, .8, .1])}})
        self.assertEqual(result["models"]["ordinal-s1"]["paired_accuracy"]["reference"], "baseline-s1")
        self.assertEqual(result["models"]["ordinal-s1"]["paired_accuracy"]["delta"], 0.)


class OrdinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        cls.mx = mx

    def test_order_distance_padding_and_non_score_rows(self):
        from lars.train.lora import ordinal_loss
        mx = self.mx
        target = mx.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        mask = mx.array([[True, True, True, False], [True, True, False, False]])
        scores = mx.array([True, False])
        near = mx.array([[0., 1., 0., 0.], [1., 0., 0., 0.]])
        far = mx.array([[0., 0., 1., 0.], [0., 1., 0., 0.]])
        self.assertAlmostEqual(float(ordinal_loss(mx, target, target, mask, scores)), 0.)
        self.assertAlmostEqual(float(ordinal_loss(mx, near, target, mask, scores)), .5)
        self.assertAlmostEqual(float(ordinal_loss(mx, far, target, mask, scores)), 1.)
        self.assertEqual(float(ordinal_loss(mx, far, target, mask, mx.array([False, False]))), 0.)

    def test_gradient_is_finite_and_points_toward_gold(self):
        from lars.train.lora import ordinal_loss
        mx = self.mx
        target = mx.array([[1., 0., 0.]])
        fn = lambda z: ordinal_loss(mx, mx.softmax(z), target, mx.array([[True]*3]), mx.array([True]))
        grad = np.asarray(mx.grad(fn)(mx.array([[0., 0., 0.]])))
        self.assertTrue(np.isfinite(grad).all())
        self.assertLess(grad[0, 0], 0.)
        self.assertGreater(grad[0, 2], 0.)

    def test_shared_head_matches_numpy_serving_and_option_permutation(self):
        from lars.heads import PointerHeadScorer
        from lars.train.residual import build_head
        mx = self.mx
        mx.random.seed(4)
        head = build_head(8, 4)
        head.q.weight = mx.random.normal((4, 8)) * .1
        head.log_s = mx.array([.1, .2, .3])
        head.bias = mx.array([.4, .0, .0])
        h, options, z = mx.random.normal((1, 8)), mx.random.normal((1, 3, 8)), mx.array([[.3, -.4, .5]])
        kind, mask = mx.array([0]), mx.array([[True]*3])
        actual = np.asarray(head(h, options, z, mask, kind))[0]
        scorer = PointerHeadScorer(np.asarray(head.q.weight), np.asarray(head.k.weight),
                                   np.asarray(head.log_s), np.asarray(head.bias), np.eye(8))
        expected = scorer.adjust(np.asarray(z)[0], np.asarray(h)[0], np.asarray(options)[0], "noul")
        # Metal's matrix products use reduced internal precision; numpy is the CPU reference.
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=3e-4)
        # Choice has no position-specific bias. Realigning shuffled options must agree.
        order = mx.array([2, 0, 1])
        original = np.asarray(head(h, options, z, mask, mx.array([1])))[0]
        shuffled = np.asarray(head(h, options[:, order], z[:, order], mask, mx.array([1])))[0]
        np.testing.assert_allclose(shuffled, original[[2, 0, 1]], rtol=1e-5, atol=1e-5)

    def test_feature_logits_keep_the_models_bfloat16_projection(self):
        from lars.backends.mlx import MLXBackend
        mx = self.mx
        backend = MLXBackend.__new__(MLXBackend)
        backend._mx = mx
        backend.model = object()
        backend._encode = lambda text: [1, 2] if text else []
        backend._label_id = {"A": 0, "B": 1}.get
        backend._make_cache = lambda model: None
        backend._hidden_forward = lambda ids, cache: mx.array([[0., 0.], [.999, 1.001]], mx.bfloat16)
        backend._text = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False), lm_head=lambda h: h)
        backend.token_offsets = lambda text: [(0, 1), (1, 2)]
        backend.label_rows = lambda labels: self.fail("feature readout must use the model projection")
        z, _, _ = backend.label_logits_with_features("", ["xy"], [("A", "B")], [(1, 2)])[0]
        np.testing.assert_array_equal(z, [1., 1.])


if __name__ == "__main__":
    unittest.main()
