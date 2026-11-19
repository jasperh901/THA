from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BackboneConfig
from .modules import LayerwiseFusion


def resolve_module(root: nn.Module, dotted_path: str) -> Any:
    current: Any = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            raise AttributeError(
                f"Cannot resolve '{dotted_path}': {type(current).__name__} has no '{part}'"
            )
        current = getattr(current, part)
    return current


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


@dataclass
class RelayState:
    memory: Tensor
    audio_reliability: Tensor
    vision_reliability: Tensor
    fusion_span: tuple[int, int]
    batch_size: int


class LayerInjectionManager:
    """Applies THA updates through hooks without copying or editing backbone code."""

    def __init__(
        self,
        model: nn.Module,
        layer_path: str,
        selected_layers: tuple[int, ...],
        fusion: LayerwiseFusion,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.fusion = fusion
        self.enabled = enabled
        layers = resolve_module(model, layer_path)
        if not isinstance(layers, (nn.ModuleList, list, tuple)):
            raise TypeError(f"Resolved layer container '{layer_path}' is not an indexed sequence")
        if max(selected_layers) >= len(layers):
            raise ValueError(
                f"Selected layer {max(selected_layers)} exceeds backbone depth {len(layers)}"
            )
        self.state: RelayState | None = None
        self.handles = [
            layers[index].register_forward_hook(self._make_hook(index)) for index in selected_layers
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @contextmanager
    def activate(self, state: RelayState | None) -> Iterator[None]:
        previous = self.state
        self.state = state
        try:
            yield
        finally:
            self.state = previous

    def _batch_first(self, hidden: Tensor, state: RelayState) -> tuple[Tensor, bool] | None:
        if hidden.ndim != 3:
            return None
        end = state.fusion_span[1]
        if hidden.size(0) == state.batch_size and hidden.size(1) >= end:
            return hidden, False
        if hidden.size(1) == state.batch_size and hidden.size(0) >= end:
            return hidden.transpose(0, 1), True
        return None

    @staticmethod
    def _replace_hidden(output: Any, hidden: Tensor) -> Any:
        if torch.is_tensor(output):
            return hidden
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        if isinstance(output, list):
            return [hidden, *output[1:]]
        return output

    def _make_hook(self, layer_index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            state = self.state
            if state is None or not self.enabled:
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden):
                return output
            normalized = self._batch_first(hidden, state)
            if normalized is None:
                return output
            batch_first, was_transposed = normalized
            updated = self.fusion(
                batch_first,
                state.memory,
                state.audio_reliability,
                state.vision_reliability,
                state.fusion_span,
                layer_index,
            )
            if was_transposed:
                updated = updated.transpose(0, 1)
            return self._replace_hidden(output, updated)

        return hook


class FrozenBackbone(nn.Module):
    def __init__(
        self,
        config: BackboneConfig,
        model: nn.Module,
        tokenizer: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        config: BackboneConfig,
        model_name_or_path: str | None = None,
        device: str | torch.device = "cuda",
    ) -> "FrozenBackbone":
        source = model_name_or_path or config.model_id
        dtype = torch_dtype(config.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=config.trust_remote_code,
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=dtype,
        )
        model.to(device)
        return cls(config, model, tokenizer)

    def train(self, mode: bool = True) -> "FrozenBackbone":
        super().train(False)
        self.model.eval()
        return self

    @property
    def embedding(self) -> nn.Module:
        embedding = self.model.get_input_embeddings()
        if embedding is None:
            raise RuntimeError("The backbone does not expose an input embedding table")
        return embedding

    def embed(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids)

    def forward(self, **kwargs: Any) -> Any:
        return self.model(**kwargs)
