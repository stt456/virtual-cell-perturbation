"""Regression: equivalent array layouts must not invalidate a frozen mean."""
import unittest
import numpy as np
from scripts.evaluate_c_results import validated_training_mean


class MeanPrecisionChecks(unittest.TestCase):
    def test_equal_population_with_different_reduction_layout(self):
        values = np.random.default_rng(13).normal(size=(760, 25))
        c_mean = np.array(values, order="C").mean(axis=0)
        f_mean = np.array(values, order="F").mean(axis=0)
        self.assertGreater(np.max(np.abs(c_mean-f_mean)), 0)
        np.testing.assert_array_equal(validated_training_mean(f_mean, c_mean), f_mean)

    def test_different_population_mean_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "training equal-weight mean"):
            validated_training_mean(np.array([.1, .2]), np.array([.1, .201]))


if __name__ == "__main__":
    unittest.main()
