from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from .backbone import FrozenBackbone, LayerInjectionManager, RelayState
from .config import RunConfig
from .metrics import decode_prediction
from .modules import THAOutput, ThalamusAdapter


class THAModel(nn.Module):
    """THA wrapped around an unmodified, frozen Hugging Face causal language model."""

    def __init__(
        self,
        config: RunConfig,
        backbone: FrozenBackbone,
        disabled_modules: Iterable[str] = (),
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.disabled_modules = frozenset(disabled_modules)
        self.backbone.tokenizer.padding_side = "left"
        anchor_embeddings = self._build_anchor_bank(config.dataset.anchors)
        self.adapter = ThalamusAdapter(
            text_dim=config.backbone.hidden_size,
            audio_dim=config.dataset.audio_dim,
            vision_dim=config.dataset.vision_dim,
            hidden_dim=config.adapter.hidden_dim,
            rank=config.adapter.rank,
            evidence_tokens=config.adapter.evidence_tokens,
            affective_tokens=config.adapter.affective_tokens,
            attention_heads=config.adapter.attention_heads,
            selected_layers=config.backbone.selected_layers,
            roles=config.dataset.roles,
            anchor_embeddings=anchor_embeddings,
            expert_bottleneck=config.adapter.expert_bottleneck,
            expert_temperature=config.adapter.expert_temperature,
            expert_residual_gate_bias=config.adapter.expert_residual_gate_bias,
            use_reliability_layer_gate=config.adapter.use_reliability_layer_gate,
            dropout=config.adapter.dropout,
            disabled_modules=self.disabled_modules,
        )
        self.adapter.to(self.device)
        self.injection = LayerInjectionManager(
            self.backbone.model,
            config.backbone.layer_path,
            config.backbone.selected_layers,
            self.adapter.layerwise_fusion,
            enabled="layerwise_fusion" not in self.disabled_modules,
        )

    @classmethod
    def from_pretrained(
        cls,
        config: RunConfig,
        model_name_or_path: str | None = None,
        device: str | torch.device = "cuda",
        disabled_modules: Iterable[str] = (),
    ) -> "THAModel":
        backbone = FrozenBackbone.from_pretrained(config.backbone, model_name_or_path, device)
        return cls(config, backbone, disabled_modules)

    @property
    def device(self) -> torch.device:
        return self.backbone.embedding.weight.device

    @property
    def tokenizer(self) -> Any:
        return self.backbone.tokenizer

    def train(self, mode: bool = True) -> "THAModel":
        super().train(mode)
        self.backbone.train(False)
        self.adapter.train(mode)
        return self

    def _token_ids(self, text: str) -> Tensor:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def _build_anchor_bank(self, anchors: Iterable[str]) -> Tensor:
        vectors: list[Tensor] = []
        with torch.no_grad():
            for anchor in anchors:
                token_ids = self._token_ids(str(anchor))
                if token_ids.numel() == 0:
                    raise ValueError(f"Anchor '{anchor}' produced no tokens")
                vectors.append(self.backbone.embed(token_ids).float().mean(dim=1).squeeze(0))
        return torch.stack(vectors)

    def _text_inputs(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.config.dataset.max_text_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device).bool()
        with torch.no_grad():
            embeddings = self.backbone.embed(ids).detach()
        return embeddings, mask

    def _prefix(
        self,
        texts: list[str],
        audio: Tensor,
        audio_lengths: Tensor,
        vision: Tensor,
        vision_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, RelayState, THAOutput]:
        text_embeddings, text_mask = self._text_inputs(texts)
        adapter_output = self.adapter(
            text_embeddings,
            text_mask,
            audio.to(self.device),
            audio_lengths.to(self.device),
            vision.to(self.device),
            vision_lengths.to(self.device),
        )
        batch_size = text_embeddings.size(0)
        before_ids = self._token_ids(self.config.adapter.wrapper_before).expand(batch_size, -1)
        suffix_ids = self._token_ids(
            self.config.adapter.wrapper_after + self.config.dataset.prompt
        ).expand(batch_size, -1)
        with torch.no_grad():
            before = self.backbone.embed(before_ids).detach()
            suffix = self.backbone.embed(suffix_ids).detach()
        prefix = torch.cat(
            [before, text_embeddings, adapter_output.affective_tokens, suffix], dim=1
        )
        ones_before = torch.ones(before.shape[:2], dtype=torch.long, device=self.device)
        ones_affective = torch.ones(
            adapter_output.affective_tokens.shape[:2], dtype=torch.long, device=self.device
        )
        ones_suffix = torch.ones(suffix.shape[:2], dtype=torch.long, device=self.device)
        attention_mask = torch.cat(
            [ones_before, text_mask.long(), ones_affective, ones_suffix], dim=1
        )
        fusion_start = before.size(1) + text_embeddings.size(1)
        state = RelayState(
            memory=adapter_output.memory,
            audio_reliability=adapter_output.audio_reliability,
            vision_reliability=adapter_output.vision_reliability,
            fusion_span=(fusion_start, fusion_start + self.config.adapter.affective_tokens),
            batch_size=batch_size,
        )
        return prefix, attention_mask, state, adapter_output

    def _answers(self, targets: Tensor) -> tuple[Tensor, Tensor]:
        answers: list[str] = []
        if self.config.dataset.task == "regression":
            low, high = self.config.dataset.score_range
            for target in targets.detach().cpu().tolist():
                answers.append(f" {min(max(float(target), low), high):.1f}")
        else:
            for target in targets.detach().cpu().tolist():
                answers.append(" " + self.config.dataset.labels[int(target)])
        encoded = self.tokenizer(
            answers,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            eos_ids = torch.full((ids.size(0), 1), eos, dtype=ids.dtype, device=self.device)
            ids = torch.cat([ids, eos_ids], dim=1)
            mask = torch.cat([mask, torch.ones_like(eos_ids)], dim=1)
        return ids, mask

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        prefix, prefix_mask, state, adapter_output = self._prefix(
            list(batch["texts"]),
            batch["audio"],
            batch["audio_lengths"],
            batch["vision"],
            batch["vision_lengths"],
        )
        answer_ids, answer_mask = self._answers(batch["targets"])
        with torch.no_grad():
            answer_embeddings = self.backbone.embed(answer_ids).detach()
        inputs = torch.cat([prefix, answer_embeddings], dim=1)
        attention_mask = torch.cat([prefix_mask, answer_mask], dim=1)
        labels = torch.full(attention_mask.shape, -100, dtype=torch.long, device=self.device)
        labels[:, prefix.size(1) :] = answer_ids.masked_fill(answer_mask == 0, -100)
        with self.injection.activate(state):
            output = self.backbone(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
        return {
            "loss": output.loss,
            "logits": output.logits,
            "diagnostics": adapter_output.diagnostics,
        }

    @torch.no_grad()
    def generate(
        self, batch: Mapping[str, Any], max_new_tokens: int | None = None
    ) -> dict[str, Any]:
        self.eval()
        inputs, attention_mask, state, _ = self._prefix(
            list(batch["texts"]),
            batch["audio"],
            batch["audio_lengths"],
            batch["vision"],
            batch["vision_lengths"],
        )
        steps = max_new_tokens or self.config.training.max_new_tokens
        eos = self.tokenizer.eos_token_id
        finished = torch.zeros(inputs.size(0), dtype=torch.bool, device=self.device)
        generated: list[Tensor] = []
        for _ in range(steps):
            with self.injection.activate(state):
                output = self.backbone(
                    inputs_embeds=inputs,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            next_ids = output.logits[:, -1, :].argmax(dim=-1)
            if eos is not None:
                next_ids = torch.where(finished, torch.full_like(next_ids, eos), next_ids)
                finished |= next_ids.eq(eos)
            generated.append(next_ids)
            inputs = torch.cat([inputs, self.backbone.embed(next_ids[:, None])], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_ids[:, None])], dim=1)
            if bool(finished.all()):
                break
        token_ids = (
            torch.stack(generated, dim=1)
            if generated
            else torch.empty(inputs.size(0), 0, dtype=torch.long, device=self.device)
        )
        texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        predictions = [decode_prediction(text, self.config.dataset) for text in texts]
        return {"texts": texts, "predictions": predictions, "token_ids": token_ids}

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def save_adapter(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "tha-adapter-v1",
            "config": self.config.to_dict(),
            "disabled_modules": sorted(self.disabled_modules),
            "state_dict": self.adapter.state_dict(),
            "extra": dict(extra or {}),
        }
        torch.save(payload, output)
        return output

    def load_adapter(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location="cpu")
        if payload.get("format") != "tha-adapter-v1":
            raise ValueError("Unsupported checkpoint format")
        self.adapter.load_state_dict(payload["state_dict"], strict=strict)
        return dict(payload.get("extra", {}))

    def export_run_config(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
