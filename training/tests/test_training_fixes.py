"""Model-free tests for the MLX feature readout and the ordinal LoRA loss (CPU is enough)."""
import unittest
from types import SimpleNamespace

import numpy as np


class OrdinalLossTests(unittest.TestCase):
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


class FeatureReadoutTests(unittest.TestCase):
    def test_feature_logits_keep_the_models_bfloat16_projection(self):
        import mlx.core as mx
        from lars.backends.mlx import MLXBackend
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
        # In BF16 both logits round to 1.0; the old float32 recomputation would rank B above A.
        np.testing.assert_array_equal(z, [1., 1.])


if __name__ == "__main__":
    unittest.main()
