from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tha.config import compose_config
from tha.metrics import classification_metrics, decode_prediction, regression_metrics


class MetricsTest(unittest.TestCase):
    def test_regression_metrics_and_decode(self) -> None:
        config = compose_config("qwen18", "mosei", Path(tempfile.gettempdir()))
        self.assertEqual(decode_prediction("The sentiment is +1.25.", config.dataset), 1.25)
        metrics = regression_metrics([1.0, -1.0, 2.0], [1.0, -1.0, 1.0])
        self.assertAlmostEqual(metrics["MAE"], 1.0 / 3.0)
        self.assertEqual(metrics["Acc2"], 1.0)

    def test_mosei_has_zero_convention(self) -> None:
        metrics = regression_metrics([-0.1, -0.1], [0.0, -1.0])
        self.assertEqual(metrics["Acc2"], 0.5)
        self.assertEqual(metrics["Non0Acc2"], 1.0)

    def test_simsv2_weak_accuracy(self) -> None:
        metrics = regression_metrics([-0.2, 0.2, -0.7], [-0.1, 0.1, -0.8], "simsv2")
        self.assertEqual(metrics["Acc2Weak"], 1.0)

    def test_classification_metrics_and_decode(self) -> None:
        config = compose_config("qwen18", "cherma", Path(tempfile.gettempdir()))
        self.assertEqual(decode_prediction("高兴", config.dataset), 3)
        metrics = classification_metrics([0, 1, -1], [0, 1, 2])
        self.assertAlmostEqual(metrics["Acc"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["Coverage"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
