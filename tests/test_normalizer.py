import unittest

import torch

from ppo.ppo_utils import WelfordNormalizer


class NormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = WelfordNormalizer((2,))
        self.normalizer.manual_load(torch.tensor([-1., 2.]), torch.tensor([0., 8.]), 3)
        self.normalizer.eval()

    def test_constant_feature_activation_keeps_original_units(self):
        normalized = self.normalizer.normalize(torch.tensor([0., 4.]))
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(normalized[0].item(), 1.)

    def test_varying_features_keep_standard_deviation_scaling(self):
        normalized = self.normalizer.normalize(torch.tensor([-1., 6.]))
        expected = (6. - 2.) / ((8. / (3 - 1)) ** .5 + self.normalizer.eps)
        self.assertAlmostEqual(normalized[1].item(), expected)

    def test_evaluation_preserves_statistics_for_single_and_batch_inputs(self):
        mean = self.normalizer.mean.clone()
        variance_sum = self.normalizer.M2.clone()
        count = self.normalizer.count.value
        self.normalizer.normalize(torch.tensor([0., 4.]))
        batch = self.normalizer.normalize(torch.tensor([[0., 4.], [2., 6.]]))
        torch.testing.assert_close(batch, torch.tensor([[1., 1.], [3., 2.]]))
        torch.testing.assert_close(self.normalizer.mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(self.normalizer.M2, variance_sum, rtol=0, atol=0)
        self.assertEqual(self.normalizer.count.value, count)


if __name__ == '__main__':
    unittest.main()
