from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tha.config import compose_config
from tha.data import collate_samples, load_splits


class ConfigDataTest(unittest.TestCase):
    def test_compose_and_load_array_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "MOSEI" / "Processed" / "unaligned_50.pkl"
            path.parent.mkdir(parents=True)
            split = {
                "raw_text": ["clear positive affect", "negative affect"],
                "audio": np.ones((2, 3, 74), dtype=np.float32),
                "vision": np.ones((2, 2, 35), dtype=np.float32),
                "id": ["a", "b"],
                "regression_labels": np.asarray([[1.0], [-1.0]], dtype=np.float32),
            }
            with path.open("wb") as handle:
                pickle.dump({name: split for name in ("train", "valid", "test")}, handle)
            config = compose_config("qwen18", "mosei", root)
            self.assertEqual(config.seed, 7777)
            self.assertEqual(config.training.max_new_tokens, 4)
            datasets = load_splits(config.dataset)
            self.assertEqual(len(datasets["train"]), 2)
            batch = collate_samples([datasets["train"][0], datasets["train"][1]])
            self.assertEqual(tuple(batch["audio"].shape), (2, 3, 74))
            self.assertEqual(batch["targets"].tolist(), [1.0, -1.0])

    def test_best_profile_defaults_and_seed_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chat_sims = compose_config("chatglm3", "simsv2", root)
            self.assertEqual(chat_sims.seed, 2222)
            self.assertEqual(chat_sims.training.learning_rate, 5.0e-5)
            self.assertEqual(chat_sims.training.epochs, 30)

            llama_cherma = compose_config("llama2", "cherma", root, seed=9)
            self.assertEqual(llama_cherma.seed, 9)
            self.assertEqual(llama_cherma.training.learning_rate, 3.5e-5)
            self.assertEqual(llama_cherma.training.epochs, 24)
            self.assertFalse(llama_cherma.adapter.use_reliability_layer_gate)

            qwen_sims = compose_config("qwen18", "simsv2", root)
            self.assertAlmostEqual(qwen_sims.adapter.expert_temperature, 0.85)
            self.assertAlmostEqual(qwen_sims.adapter.expert_residual_gate_bias, -2.8)

            qwen_mosei = compose_config("qwen18", "mosei", root)
            self.assertEqual(qwen_mosei.adapter.expert_residual_gate_bias, 0.0)

    def test_all_audited_profile_training_defaults(self) -> None:
        expected = {
            ("qwen18", "mosei"): (7777, 4, 5.0e-5, 30, 10, 4),
            ("chatglm3", "mosei"): (3333, 1, 5.0e-5, 30, 10, 4),
            ("llama2", "mosei"): (2022, 4, 3.0e-5, 24, 10, 5),
            ("qwen18", "simsv2"): (3333, 4, 4.0e-5, 20, 10, 4),
            ("chatglm3", "simsv2"): (2222, 1, 5.0e-5, 30, 10, 4),
            ("llama2", "simsv2"): (2022, 4, 4.0e-5, 20, 10, 4),
            ("qwen18", "cherma"): (7777, 4, 5.0e-5, 30, 8, 4),
            ("chatglm3", "cherma"): (3333, 1, 5.0e-5, 30, 8, 2),
            ("llama2", "cherma"): (5055, 1, 3.5e-5, 24, 8, 4),
        }
        with tempfile.TemporaryDirectory() as directory:
            for (backbone, dataset), values in expected.items():
                config = compose_config(backbone, dataset, directory)
                observed = (
                    config.seed,
                    config.training.batch_size,
                    config.training.learning_rate,
                    config.training.epochs,
                    config.training.early_stop,
                    config.training.max_new_tokens,
                )
                self.assertEqual(observed, values, f"{backbone}/{dataset}")


if __name__ == "__main__":
    unittest.main()
