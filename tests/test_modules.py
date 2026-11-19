from __future__ import annotations

import unittest

import torch

from tha.modules import ThalamusAdapter


class ModulesTest(unittest.TestCase):
    def test_shapes_and_gradients(self) -> None:
        torch.manual_seed(7)
        adapter = ThalamusAdapter(
            text_dim=16,
            audio_dim=5,
            vision_dim=7,
            hidden_dim=8,
            rank=4,
            evidence_tokens=3,
            affective_tokens=4,
            attention_heads=2,
            selected_layers=(0, 1, 2),
            roles=("magnitude", "magnitude", "polarity", "polarity"),
            anchor_embeddings=torch.randn(6, 16),
            dropout=0.0,
        )
        output = adapter(
            torch.randn(2, 6, 16),
            torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool),
            torch.randn(2, 9, 5),
            torch.tensor([7, 9]),
            torch.randn(2, 8, 7),
            torch.tensor([8, 5]),
        )
        self.assertEqual(tuple(output.affective_tokens.shape), (2, 4, 16))
        self.assertEqual(tuple(output.memory.shape), (2, 6, 8))
        output.affective_tokens.square().mean().backward()
        gradients = [
            parameter.grad for parameter in adapter.parameters() if parameter.requires_grad
        ]
        self.assertTrue(any(value is not None for value in gradients))
        self.assertIsNone(adapter.plain_projector)

    def test_semantic_ablation_instantiates_only_plain_projector(self) -> None:
        adapter = ThalamusAdapter(
            text_dim=16,
            audio_dim=5,
            vision_dim=7,
            hidden_dim=8,
            rank=4,
            evidence_tokens=3,
            affective_tokens=4,
            attention_heads=2,
            selected_layers=(0, 1, 2),
            roles=("magnitude", "magnitude", "polarity", "polarity"),
            anchor_embeddings=torch.randn(6, 16),
            dropout=0.0,
            disabled_modules=("semantic_projector",),
        )
        self.assertIsNone(adapter.semantic_projector)
        self.assertIsNotNone(adapter.plain_projector)


if __name__ == "__main__":
    unittest.main()
