from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    display_name: str
    model_id: str
    trust_remote_code: bool
    hidden_size: int
    layer_path: str
    selected_layers: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    task: str
    format: str
    path: Path
    max_text_length: int
    audio_dim: int
    vision_dim: int
    prompt: str
    primary_metric: str
    metric_mode: str
    roles: tuple[str, ...]
    anchors: tuple[str, ...]
    labels: tuple[str, ...] = ()
    score_range: tuple[float, float] = (-1.0, 1.0)
    file_prefix: str = ""


@dataclass(frozen=True)
class AdapterConfig:
    hidden_dim: int
    rank: int
    evidence_tokens: int
    affective_tokens: int
    attention_heads: int
    dropout: float
    expert_bottleneck: int
    expert_temperature: float
    use_reliability_layer_gate: bool
    wrapper_before: str
    wrapper_after: str
    expert_residual_gate_bias: float = -3.0


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    early_stop: int
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation: int = 1
    max_new_tokens: int = 4
    num_workers: int = 0


@dataclass(frozen=True)
class RunConfig:
    backbone: BackboneConfig
    dataset: DatasetConfig
    adapter: AdapterConfig
    training: TrainingConfig
    seed: int
    output_dir: Path

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dataset"]["path"] = str(self.dataset.path)
        result["output_dir"] = str(self.output_dir)
        return result


def _config_root(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()
    return Path(__file__).resolve().parent / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def compose_config(
    backbone: str,
    dataset: str,
    data_root: str | Path,
    output_dir: str | Path = "outputs",
    seed: int | None = None,
    config_dir: str | Path | None = None,
) -> RunConfig:
    root = _config_root(config_dir)
    backbones = _read_yaml(root / "backbones.yaml")
    datasets = _read_yaml(root / "datasets.yaml")
    adapter_raw = _read_yaml(root / "adapter.yaml")

    if backbone not in backbones:
        raise KeyError(f"Unknown backbone '{backbone}'. Choose from: {', '.join(backbones)}")
    if dataset not in datasets:
        raise KeyError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(datasets)}")

    backbone_raw = backbones[backbone]
    dataset_raw = datasets[dataset]
    profile_raw = dataset_raw["profiles"].get(backbone)
    if profile_raw is None:
        raise ValueError(f"No training profile for {backbone} on {dataset}")
    profile_raw = dict(profile_raw)
    profile_seed = int(profile_raw.pop("seed"))
    dataset_overrides = profile_raw.pop("dataset", {})
    adapter_overrides = profile_raw.pop("adapter", {})
    training_raw = profile_raw.pop("training", profile_raw)
    if profile_raw and training_raw is not profile_raw:
        raise ValueError(f"Unexpected profile keys for {backbone}/{dataset}: {profile_raw}")

    effective_dataset = {**dataset_raw, **dataset_overrides}

    backbone_config = BackboneConfig(
        name=backbone,
        display_name=str(backbone_raw["display_name"]),
        model_id=str(backbone_raw["model_id"]),
        trust_remote_code=bool(backbone_raw["trust_remote_code"]),
        hidden_size=int(backbone_raw["hidden_size"]),
        layer_path=str(backbone_raw["layer_path"]),
        selected_layers=tuple(int(x) for x in backbone_raw["selected_layers"]),
        dtype=str(backbone_raw["dtype"]),
    )

    dataset_path = Path(data_root).expanduser().resolve() / effective_dataset["relative_path"]
    dataset_config = DatasetConfig(
        name=dataset,
        display_name=str(effective_dataset["display_name"]),
        task=str(effective_dataset["task"]),
        format=str(effective_dataset["format"]),
        path=dataset_path,
        file_prefix=str(effective_dataset.get("file_prefix", "")),
        max_text_length=int(effective_dataset["max_text_length"]),
        audio_dim=int(effective_dataset["audio_dim"]),
        vision_dim=int(effective_dataset["vision_dim"]),
        prompt=str(effective_dataset["prompt"]),
        primary_metric=str(effective_dataset["primary_metric"]),
        metric_mode=str(effective_dataset["metric_mode"]),
        roles=tuple(str(x) for x in effective_dataset["roles"]),
        anchors=tuple(str(x) for x in effective_dataset["anchors"]),
        labels=tuple(str(x) for x in effective_dataset.get("labels", [])),
        score_range=tuple(float(x) for x in effective_dataset.get("score_range", [-1.0, 1.0])),
    )

    adapter_config = AdapterConfig(**{**adapter_raw, **adapter_overrides})
    training_config = TrainingConfig(**training_raw)
    run_config = RunConfig(
        backbone=backbone_config,
        dataset=dataset_config,
        adapter=adapter_config,
        training=training_config,
        seed=profile_seed if seed is None else int(seed),
        output_dir=Path(output_dir).expanduser().resolve(),
    )
    validate_config(run_config)
    return run_config


def validate_config(config: RunConfig) -> None:
    if config.dataset.task not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task: {config.dataset.task}")
    if config.dataset.format not in {"array", "records"}:
        raise ValueError(f"Unsupported dataset format: {config.dataset.format}")
    if config.adapter.hidden_dim % config.adapter.attention_heads:
        raise ValueError("adapter.hidden_dim must be divisible by attention_heads")
    if len(config.dataset.roles) != config.adapter.affective_tokens:
        raise ValueError("The role list must match the affective token budget")
    if max(config.backbone.selected_layers) < 0:
        raise ValueError("Selected layer indices must be non-negative")
    if config.dataset.task == "classification" and not config.dataset.labels:
        raise ValueError("Classification datasets require a label inventory")
    if config.seed < 0:
        raise ValueError("seed must be non-negative")
