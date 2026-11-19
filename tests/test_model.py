from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tha.backbone import FrozenBackbone
from tha.config import AdapterConfig, BackboneConfig, DatasetConfig, RunConfig, TrainingConfig
from tha.model import THAModel


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    eos_token = "<eos>"
    unk_token = "<unk>"
    padding_side = "left"

    def __call__(
        self,
        text: str | list[str],
        *,
        padding: str | bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        values = [text] if isinstance(text, str) else text
        ids = [[2 + (ord(char) % 61) for char in value] or [2] for value in values]
        if truncation and max_length is not None:
            ids = [row[-max_length:] for row in ids]
        width = max_length if padding == "max_length" else max(len(row) for row in ids)
        padded, masks = [], []
        for row in ids:
            pad = [0] * (width - len(row))
            padded.append(pad + row)
            masks.append([0] * len(pad) + [1] * len(row))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def batch_decode(self, ids: torch.Tensor, **_: Any) -> list[str]:
        return ["0.0" for _ in range(ids.size(0))]


class FakeLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.tanh(self.linear(hidden)),)


class FakeCausalLM(nn.Module):
    def __init__(self, width: int = 16, vocab: int = 80) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([FakeLayer(width) for _ in range(3)])
        self.output = nn.Linear(width, vocab)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, inputs_embeds: torch.Tensor, labels=None, **_: Any) -> SimpleNamespace:
        hidden = inputs_embeds
        for layer in self.transformer.h:
            hidden = layer(hidden)[0]
        logits = self.output(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def fake_config(root: Path) -> RunConfig:
    return RunConfig(
        backbone=BackboneConfig(
            "fake", "Fake", "fake", False, 16, "transformer.h", (0, 1, 2), "float32"
        ),
        dataset=DatasetConfig(
            "fake",
            "Fake",
            "regression",
            "array",
            root,
            12,
            5,
            7,
            "Score:",
            "MAE",
            "min",
            ("magnitude", "magnitude", "polarity", "polarity"),
            ("positive", "negative", "neutral"),
            score_range=(-3.0, 3.0),
        ),
        adapter=AdapterConfig(8, 4, 3, 4, 2, 0.0, 8, 1.0, True, "<M>", "</M>"),
        training=TrainingConfig(2, 1e-4, 1, 1, max_new_tokens=2),
        seed=1,
        output_dir=root,
    )


class ModelTest(unittest.TestCase):
    def test_teacher_forcing_generation_and_adapter_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backbone = FrozenBackbone(fake_config(root).backbone, FakeCausalLM(), FakeTokenizer())
            model = THAModel(fake_config(root), backbone)
            batch = {
                "texts": ["good", "bad"],
                "audio": torch.randn(2, 6, 5),
                "vision": torch.randn(2, 5, 7),
                "audio_lengths": torch.tensor([6, 4]),
                "vision_lengths": torch.tensor([5, 3]),
                "targets": torch.tensor([1.0, -1.0]),
            }
            output = model(batch)
            self.assertTrue(torch.isfinite(output["loss"]))
            output["loss"].backward()
            self.assertTrue(any(p.grad is not None for p in model.adapter.parameters()))
            generated = model.generate(batch)
            self.assertEqual(generated["predictions"], [0.0, 0.0])
            path = model.save_adapter(root / "adapter.pt")
            self.assertLess(path.stat().st_size, 2_000_000)
            model.load_adapter(path)


if __name__ == "__main__":
    unittest.main()
