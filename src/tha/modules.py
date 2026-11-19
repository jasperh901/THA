from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def lengths_to_mask(lengths: Tensor, max_length: int) -> Tensor:
    lengths = lengths.to(dtype=torch.long).clamp(min=1, max=max_length)
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def masked_mean(values: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class MaskAwareTextSummary(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(text_dim)
        self.projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, text_embeddings: Tensor, text_mask: Tensor) -> Tensor:
        pooled = masked_mean(text_embeddings, text_mask)
        return self.projection(self.norm(pooled))


class TemporalEvidenceEncoder(nn.Module):
    """Selects a fixed evidence bank from one variable-length sensory stream."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        evidence_tokens: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.evidence_tokens = evidence_tokens
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.time_projection = nn.Linear(4, hidden_dim)
        self.modality_bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.temporal_convolutions = nn.ModuleList(
            nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel // 2, groups=hidden_dim)
            for kernel in (3, 5, 9)
        )
        self.temporal_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.memory_queries = nn.Parameter(torch.empty(evidence_tokens, hidden_dim))
        self.context_to_query = nn.Linear(hidden_dim, hidden_dim)
        self.reliability = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.memory_queries, std=0.02)

    def _time_features(self, inputs: Tensor, lengths: Tensor) -> Tensor:
        batch, sequence_length, _ = inputs.shape
        positions = (
            torch.arange(sequence_length, device=inputs.device, dtype=inputs.dtype)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        denominators = lengths.to(device=inputs.device, dtype=inputs.dtype)
        denominators = denominators.unsqueeze(1).clamp_min(1.0)
        relative = positions / denominators
        occupancy = denominators.expand_as(relative) / max(sequence_length, 1)
        return torch.stack(
            [relative, torch.sin(math.pi * relative), torch.cos(math.pi * relative), occupancy],
            dim=-1,
        )

    def forward(
        self, inputs: Tensor, lengths: Tensor, text_context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if inputs.ndim != 3:
            raise ValueError(f"Expected [batch, time, width], received {tuple(inputs.shape)}")
        mask = lengths_to_mask(lengths, inputs.size(1))
        projected = self.input_projection(inputs)
        projected = (
            projected
            + self.time_projection(self._time_features(inputs, lengths))
            + self.modality_bias
        )
        channels_first = projected.transpose(1, 2)
        local = torch.stack(
            [convolution(channels_first) for convolution in self.temporal_convolutions], dim=0
        ).mean(dim=0)
        hidden = projected + self.temporal_mix(local.transpose(1, 2))
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)

        queries = self.memory_queries.unsqueeze(0).expand(inputs.size(0), -1, -1)
        queries = queries + self.context_to_query(text_context).unsqueeze(1)
        logits = torch.matmul(queries, hidden.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        logits = logits.masked_fill(~mask.unsqueeze(1), torch.finfo(logits.dtype).min)
        attention = F.softmax(logits, dim=-1)
        evidence = torch.matmul(attention, hidden)
        pooled = masked_mean(hidden, mask)
        reliability = torch.sigmoid(self.reliability(torch.cat([pooled, text_context], dim=-1)))
        return evidence, reliability, attention


class AffectiveQueryResampler(nn.Module):
    """Compresses two reliability-weighted evidence banks into K affective states."""

    def __init__(
        self,
        hidden_dim: int,
        affective_tokens: int,
        attention_heads: int,
        role_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(affective_tokens, hidden_dim))
        self.context_to_query = nn.Linear(hidden_dim, hidden_dim)
        self.audio_tag = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.vision_tag = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.role_embeddings = nn.Embedding(max(role_count, 1), hidden_dim)
        self.role_query_gates = nn.Parameter(torch.full((max(role_count, 1), 1), -2.5))
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.queries, std=0.02)

    def forward(
        self,
        text_context: Tensor,
        audio_evidence: Tensor,
        vision_evidence: Tensor,
        audio_reliability: Tensor,
        vision_reliability: Tensor,
        role_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        audio = (audio_evidence + self.audio_tag) * audio_reliability.unsqueeze(-1)
        vision = (vision_evidence + self.vision_tag) * vision_reliability.unsqueeze(-1)
        memory = torch.cat([audio, vision], dim=1)
        queries = self.queries.unsqueeze(0).expand(text_context.size(0), -1, -1)
        queries = queries + self.context_to_query(text_context).unsqueeze(1)
        role_delta = self.role_embeddings(role_ids).unsqueeze(0).to(queries.dtype)
        role_gate = torch.sigmoid(self.role_query_gates[role_ids]).unsqueeze(0)
        queries = queries + role_gate.to(queries.dtype) * role_delta
        retrieved, attention = self.attention(
            self.norm1(queries), memory, memory, need_weights=True, average_attn_weights=True
        )
        hidden = queries + retrieved
        hidden = hidden + self.feed_forward(self.norm2(hidden))
        return hidden, memory, attention


class ExpertGate(nn.Module):
    """Reconciles local, memory-grounded, and global affective views."""

    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int,
        bottleneck: int,
        temperature: float,
        dropout: float,
        role_count: int,
        residual_gate_bias: float,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.local = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_dim),
        )
        self.semantic_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.semantic = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_dim),
        )
        self.global_view = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 2),
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.route = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 2),
            nn.Linear(hidden_dim * 3 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.role_embeddings = nn.Embedding(max(role_count, 1), hidden_dim)
        self.role_gate_bias = nn.Parameter(torch.zeros(max(role_count, 1), 3))
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        nn.init.constant_(self.residual_gate[1].bias, residual_gate_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: Tensor,
        memory: Tensor,
        text_context: Tensor,
        audio_reliability: Tensor,
        vision_reliability: Tensor,
        role_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm(states)
        local = self.local(normalized)
        semantic_context = self.context_norm(memory)
        semantic, _ = self.semantic_attention(normalized, semantic_context, semantic_context)
        semantic = self.semantic(semantic)
        reliability = torch.cat([audio_reliability, vision_reliability], dim=-1)
        global_state = self.global_view(
            torch.cat([text_context, memory.mean(dim=1), reliability], dim=-1)
        )
        global_tokens = global_state.unsqueeze(1).expand_as(states)
        role_ids = role_ids.to(device=states.device, dtype=torch.long)
        role_features = self.role_embeddings(role_ids).unsqueeze(0).expand(states.size(0), -1, -1)
        reliability_tokens = reliability.unsqueeze(1).expand(-1, states.size(1), -1)
        routing_features = torch.cat(
            [normalized, global_tokens, role_features, reliability_tokens], dim=-1
        )
        route_logits = self.route(routing_features) + self.role_gate_bias[role_ids].unsqueeze(0)
        weights = F.softmax(route_logits / self.temperature, dim=-1)
        experts = torch.stack([local, semantic, global_tokens], dim=-2)
        update = (weights.unsqueeze(-1) * experts).sum(dim=-2)
        output = states + self.residual_gate(normalized) * self.dropout(update)
        return output, weights


class SemanticProjector(nn.Module):
    """Places compact states near frozen verbalizer anchors with a low-rank residual."""

    def __init__(
        self,
        hidden_dim: int,
        llm_dim: int,
        rank: int,
        anchor_embeddings: Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        if anchor_embeddings.ndim != 2 or anchor_embeddings.size(1) != llm_dim:
            raise ValueError("anchor_embeddings must have shape [anchors, llm_dim]")
        self.register_buffer("anchors", anchor_embeddings.detach().float(), persistent=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.anchor_weights = nn.Linear(hidden_dim, anchor_embeddings.size(0))
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, llm_dim, bias=False)
        self.residual_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: Tensor) -> Tensor:
        hidden = self.norm(states.float())
        anchor_mix = F.softmax(self.anchor_weights(hidden), dim=-1)
        semantic_base = anchor_mix @ self.anchors
        residual = self.up(self.down(self.dropout(hidden)))
        return semantic_base + self.residual_gate(hidden) * residual


class PlainLowRankProjector(nn.Module):
    def __init__(self, hidden_dim: int, llm_dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, llm_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(llm_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: Tensor) -> Tensor:
        return self.up(self.down(self.dropout(self.norm(states.float())))) + self.bias


class LayerwiseFusion(nn.Module):
    """Re-reads compact sensory memory at selected frozen Transformer depths."""

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int,
        rank: int,
        attention_heads: int,
        selected_layers: Iterable[int],
        roles: Iterable[str],
        use_reliability_gate: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.selected_layers = tuple(int(index) for index in selected_layers)
        self.roles = tuple(str(role) for role in roles)
        self.norm = nn.LayerNorm(llm_dim)
        self.to_adapter = nn.Linear(llm_dim, hidden_dim, bias=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, llm_dim, bias=False)
        self.layer_gates = nn.ParameterDict(
            {str(index): nn.Parameter(torch.tensor(-2.0)) for index in self.selected_layers}
        )
        self.reliability_gate = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim + 2),
                nn.Linear(hidden_dim + 2, max(hidden_dim // 2, 16)),
                nn.GELU(),
                nn.Linear(max(hidden_dim // 2, 16), 1),
            )
            if use_reliability_gate
            else None
        )

    def route_mask(self, layer_index: int, device: torch.device) -> Tensor:
        roles = self.roles
        specialized = any(role in {"magnitude", "polarity", "weak"} for role in roles)
        if not specialized or len(self.selected_layers) < 2:
            return torch.ones(len(roles), dtype=torch.bool, device=device)
        position = self.selected_layers.index(layer_index)
        if position == 0:
            mask = [role == "magnitude" for role in roles]
        elif position == len(self.selected_layers) - 1:
            mask = [role in {"weak", "polarity"} for role in roles]
        else:
            mask = [True] * len(roles)
        if not any(mask):
            mask = [True] * len(roles)
        return torch.tensor(mask, dtype=torch.bool, device=device)

    def forward(
        self,
        hidden_states: Tensor,
        memory: Tensor,
        audio_reliability: Tensor,
        vision_reliability: Tensor,
        fusion_span: tuple[int, int],
        layer_index: int,
    ) -> Tensor:
        if layer_index not in self.selected_layers:
            return hidden_states
        start, end = fusion_span
        if end > hidden_states.size(1) or end - start != len(self.roles):
            return hidden_states
        fusion = hidden_states[:, start:end, :]
        route = self.route_mask(layer_index, fusion.device)
        routed = fusion[:, route, :].float()
        query = self.to_adapter(self.norm(routed))
        retrieved, _ = self.attention(query, memory.float(), memory.float(), need_weights=False)
        retrieved = retrieved + self.feed_forward(retrieved)
        update = self.up(self.down(retrieved))
        gate_logit = self.layer_gates[str(layer_index)].to(update.dtype)
        if self.reliability_gate is not None:
            reliability = torch.cat([audio_reliability, vision_reliability], dim=-1)
            reliability = reliability.unsqueeze(1).expand(-1, retrieved.size(1), -1)
            token_logit = self.reliability_gate(torch.cat([retrieved, reliability], dim=-1))
            gate = torch.sigmoid(token_logit + gate_logit)
        else:
            gate = torch.sigmoid(gate_logit)
        updated = fusion.clone()
        updated[:, route, :] = routed.to(update.dtype) + gate * update
        result = hidden_states.clone()
        result[:, start:end, :] = updated.to(hidden_states.dtype)
        return result


@dataclass
class THAOutput:
    affective_tokens: Tensor
    memory: Tensor
    audio_reliability: Tensor
    vision_reliability: Tensor
    diagnostics: dict[str, Tensor]


class ThalamusAdapter(nn.Module):
    """The five-module THA path, independent of any specific LLM implementation."""

    MODULE_NAMES = {
        "text_guidance",
        "query_resampler",
        "expert_gate",
        "semantic_projector",
        "layerwise_fusion",
    }

    def __init__(
        self,
        *,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        hidden_dim: int,
        rank: int,
        evidence_tokens: int,
        affective_tokens: int,
        attention_heads: int,
        selected_layers: Iterable[int],
        roles: Iterable[str],
        anchor_embeddings: Tensor,
        expert_bottleneck: int = 64,
        expert_temperature: float = 1.0,
        expert_residual_gate_bias: float = -3.0,
        use_reliability_layer_gate: bool = True,
        dropout: float = 0.1,
        disabled_modules: Iterable[str] = (),
    ) -> None:
        super().__init__()
        self.disabled_modules = frozenset(disabled_modules)
        unknown = self.disabled_modules - self.MODULE_NAMES
        if unknown:
            raise ValueError(f"Unknown disabled modules: {sorted(unknown)}")
        self.hidden_dim = hidden_dim
        self.affective_tokens = affective_tokens
        self.text_summary = MaskAwareTextSummary(text_dim, hidden_dim, dropout)
        self.null_context = nn.Parameter(torch.zeros(1, hidden_dim))
        self.audio_encoder = TemporalEvidenceEncoder(
            audio_dim, hidden_dim, evidence_tokens, attention_heads, dropout
        )
        self.vision_encoder = TemporalEvidenceEncoder(
            vision_dim, hidden_dim, evidence_tokens, attention_heads, dropout
        )
        unique_roles = tuple(dict.fromkeys(str(role) for role in roles))
        self.role_to_id = {role: index for index, role in enumerate(unique_roles)}
        role_ids = [self.role_to_id[str(role)] for role in roles]
        self.register_buffer("role_ids", torch.tensor(role_ids, dtype=torch.long), persistent=True)
        self.resampler = AffectiveQueryResampler(
            hidden_dim, affective_tokens, attention_heads, len(unique_roles), dropout
        )
        self.expert_gate = ExpertGate(
            hidden_dim,
            attention_heads,
            expert_bottleneck,
            expert_temperature,
            dropout,
            len(unique_roles),
            expert_residual_gate_bias,
        )
        if "semantic_projector" in self.disabled_modules:
            self.semantic_projector = None
            self.plain_projector = PlainLowRankProjector(hidden_dim, text_dim, rank, dropout)
        else:
            self.semantic_projector = SemanticProjector(
                hidden_dim, text_dim, rank, anchor_embeddings, dropout
            )
            self.plain_projector = None
        self.layerwise_fusion = LayerwiseFusion(
            text_dim,
            hidden_dim,
            rank,
            attention_heads,
            selected_layers,
            roles,
            use_reliability_layer_gate,
            dropout,
        )

    def _fixed_pool(
        self,
        audio: Tensor,
        vision: Tensor,
        audio_reliability: Tensor,
        vision_reliability: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        audio = audio * audio_reliability.unsqueeze(-1)
        vision = vision * vision_reliability.unsqueeze(-1)
        memory = torch.cat([audio, vision], dim=1)
        pooled = F.adaptive_avg_pool1d(memory.transpose(1, 2), self.affective_tokens).transpose(
            1, 2
        )
        empty_attention = memory.new_zeros(memory.size(0), self.affective_tokens, memory.size(1))
        return pooled, memory, empty_attention

    def forward(
        self,
        text_embeddings: Tensor,
        text_mask: Tensor,
        audio: Tensor,
        audio_lengths: Tensor,
        vision: Tensor,
        vision_lengths: Tensor,
    ) -> THAOutput:
        text_work = text_embeddings.float()
        context = self.text_summary(text_work, text_mask)
        if "text_guidance" in self.disabled_modules:
            context = self.null_context.expand(text_embeddings.size(0), -1)
        audio_evidence, rho_audio, audio_attention = self.audio_encoder(
            audio.float(), audio_lengths, context
        )
        vision_evidence, rho_vision, vision_attention = self.vision_encoder(
            vision.float(), vision_lengths, context
        )
        if "query_resampler" in self.disabled_modules:
            states, memory, query_attention = self._fixed_pool(
                audio_evidence, vision_evidence, rho_audio, rho_vision
            )
        else:
            states, memory, query_attention = self.resampler(
                context,
                audio_evidence,
                vision_evidence,
                rho_audio,
                rho_vision,
                self.role_ids,
            )
        expert_weights = states.new_zeros(states.size(0), states.size(1), 3)
        if "expert_gate" not in self.disabled_modules:
            states, expert_weights = self.expert_gate(
                states, memory, context, rho_audio, rho_vision, self.role_ids
            )
        if "semantic_projector" in self.disabled_modules:
            assert self.plain_projector is not None
            tokens = self.plain_projector(states)
        else:
            assert self.semantic_projector is not None
            tokens = self.semantic_projector(states)
        return THAOutput(
            affective_tokens=tokens.to(text_embeddings.dtype),
            memory=memory,
            audio_reliability=rho_audio,
            vision_reliability=rho_vision,
            diagnostics={
                "audio_attention": audio_attention,
                "vision_attention": vision_attention,
                "query_attention": query_attention,
                "expert_weights": expert_weights,
                "text_context": context,
            },
        )
