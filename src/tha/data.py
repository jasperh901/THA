from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig


SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str
    audio: np.ndarray
    vision: np.ndarray
    audio_length: int
    vision_length: int
    target: float | int


def _finite_feature(value: Any, width: int, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[-1] != width:
        raise ValueError(f"{field} must have shape [time, {width}], received {array.shape}")
    if np.isfinite(array).all():
        return array
    return np.nan_to_num(array, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _sequence_length(array: np.ndarray, supplied: Any = None) -> int:
    if supplied is not None:
        return int(np.clip(int(supplied), 1, array.shape[0]))
    active = np.any(np.abs(array) > 0, axis=-1)
    indices = np.flatnonzero(active)
    return int(indices[-1] + 1) if indices.size else 1


def _regression_labels(split: Mapping[str, Any]) -> np.ndarray:
    preferred = ("regression_labels", "M_labels", "labels", "regression_label")
    for key in preferred:
        if key in split:
            return np.asarray(split[key], dtype=np.float32).reshape(-1)
    candidates = [key for key in split if key.endswith("_labels") and "_T" not in key]
    if len(candidates) == 1:
        return np.asarray(split[candidates[0]], dtype=np.float32).reshape(-1)
    raise KeyError(f"Cannot identify regression labels. Available fields: {sorted(split)}")


class MultimodalDataset(Dataset[Sample]):
    def __init__(self, samples: Sequence[Sample], name: str, split: str) -> None:
        self.samples = list(samples)
        self.name = name
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def _array_samples(raw: Mapping[str, Any], config: DatasetConfig, split: str) -> list[Sample]:
    if split not in raw or not isinstance(raw[split], Mapping):
        raise KeyError(f"Array dataset does not contain split '{split}'")
    values = raw[split]
    texts = values.get("raw_text")
    if texts is None:
        raise KeyError(f"{split} is missing raw_text")
    audio = values.get("audio")
    vision = values.get("vision")
    if audio is None or vision is None:
        raise KeyError(f"{split} must contain audio and vision")
    targets = _regression_labels(values)
    ids = values.get("id", np.arange(len(targets)))
    audio_lengths = values.get("audio_lengths")
    vision_lengths = values.get("vision_lengths")
    if not (len(texts) == len(audio) == len(vision) == len(targets)):
        raise ValueError(f"{split} arrays have inconsistent sample counts")

    samples: list[Sample] = []
    for index in range(len(targets)):
        audio_item = _finite_feature(audio[index], config.audio_dim, "audio")
        vision_item = _finite_feature(vision[index], config.vision_dim, "vision")
        supplied_audio = audio_lengths[index] if audio_lengths is not None else None
        supplied_vision = vision_lengths[index] if vision_lengths is not None else None
        samples.append(
            Sample(
                sample_id=str(ids[index]),
                text=str(texts[index]),
                audio=audio_item,
                vision=vision_item,
                audio_length=_sequence_length(audio_item, supplied_audio),
                vision_length=_sequence_length(vision_item, supplied_vision),
                target=float(targets[index]),
            )
        )
    return samples


def _record_samples(rows: Any, config: DatasetConfig, split: str) -> list[Sample]:
    if not isinstance(rows, Sequence):
        raise TypeError(f"{split} record file must contain a sequence")
    label_to_index = {label: index for index, label in enumerate(config.labels)}
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("features"), Mapping):
            raise TypeError(f"{split}[{index}] is not a multimodal record")
        features = row["features"]
        audio = _finite_feature(features.get("audio"), config.audio_dim, "audio")
        vision = _finite_feature(features.get("video"), config.vision_dim, "video")
        label = str(row.get("label", "")).strip()
        if label not in label_to_index:
            raise ValueError(f"{split}[{index}] has unknown label '{label}'")
        canonical = row.get("tha_metadata")
        text = features.get("text", "")
        sample_id = f"{split}:{index}"
        if isinstance(canonical, Mapping):
            text = canonical.get("target_context_text", text)
            sample_id = str(canonical.get("sample_id", sample_id))
        samples.append(
            Sample(
                sample_id=sample_id,
                text=str(text),
                audio=audio,
                vision=vision,
                audio_length=_sequence_length(audio, features.get("audio_len")),
                vision_length=_sequence_length(vision, features.get("video_len")),
                target=label_to_index[label],
            )
        )
    return samples


def load_splits(
    config: DatasetConfig, splits: Iterable[str] = SPLITS
) -> dict[str, MultimodalDataset]:
    requested = tuple(splits)
    unknown = set(requested) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    datasets: dict[str, MultimodalDataset] = {}
    if config.format == "array":
        if not config.path.is_file():
            raise FileNotFoundError(config.path)
        with config.path.open("rb") as handle:
            raw = pickle.load(handle)
        for split in requested:
            datasets[split] = MultimodalDataset(
                _array_samples(raw, config, split), config.name, split
            )
        return datasets

    if not config.path.is_dir():
        raise FileNotFoundError(config.path)
    for split in requested:
        path = config.path / f"{config.file_prefix}_{split}.pkl"
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        datasets[split] = MultimodalDataset(
            _record_samples(rows, config, split), config.name, split
        )
    return datasets


def collate_samples(samples: Sequence[Sample]) -> dict[str, Any]:
    audio = pad_sequence([torch.from_numpy(item.audio) for item in samples], batch_first=True)
    vision = pad_sequence([torch.from_numpy(item.vision) for item in samples], batch_first=True)
    regression = isinstance(samples[0].target, float)
    target_dtype = torch.float32 if regression else torch.long
    return {
        "ids": [item.sample_id for item in samples],
        "texts": [item.text for item in samples],
        "audio": audio,
        "vision": vision,
        "audio_lengths": torch.tensor([item.audio_length for item in samples], dtype=torch.long),
        "vision_lengths": torch.tensor([item.vision_length for item in samples], dtype=torch.long),
        "targets": torch.tensor([item.target for item in samples], dtype=target_dtype),
    }


def make_loaders(
    datasets: Mapping[str, MultimodalDataset],
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, DataLoader[Any]]:
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            collate_fn=collate_samples,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }


def describe_splits(datasets: Mapping[str, MultimodalDataset]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, dataset in datasets.items():
        targets = np.asarray([sample.target for sample in dataset.samples])
        report[split] = {
            "samples": len(dataset),
            "audio_shape": list(dataset.samples[0].audio.shape) if dataset.samples else None,
            "vision_shape": list(dataset.samples[0].vision.shape) if dataset.samples else None,
            "target_min": float(targets.min()) if targets.size else None,
            "target_max": float(targets.max()) if targets.size else None,
        }
    return report


def print_description(datasets: Mapping[str, MultimodalDataset]) -> None:
    print(json.dumps(describe_splits(datasets), indent=2, ensure_ascii=False))
