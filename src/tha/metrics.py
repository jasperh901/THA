from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import DatasetConfig


NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def decode_prediction(text: str, config: DatasetConfig) -> float | int:
    normalized = str(text).strip()
    if config.task == "regression":
        match = NUMBER.search(normalized.replace(",", "."))
        if match is None:
            return float("nan")
        low, high = config.score_range
        return float(np.clip(float(match.group()), low, high))
    lowered = normalized.casefold()
    for index, label in enumerate(config.labels):
        if label.casefold() == lowered or label.casefold() in lowered:
            return index
    return -1


def _bucketize(values: np.ndarray, boundaries: list[float]) -> np.ndarray:
    return np.digitize(values, boundaries, right=True)


def regression_metrics(
    predictions: Iterable[float],
    targets: Iterable[float],
    dataset_name: str = "mosei",
) -> dict[str, float]:
    pred = np.asarray(list(predictions), dtype=np.float64)
    true = np.asarray(list(targets), dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(true)
    coverage = float(valid.mean()) if valid.size else 0.0
    pred, true = pred[valid], true[valid]
    if not true.size:
        return {"MAE": float("inf"), "Corr": 0.0, "Acc2": 0.0, "F1": 0.0, "Coverage": coverage}
    mae = float(np.mean(np.abs(pred - true)))
    variable = true.size > 1 and np.std(pred) > 0 and np.std(true) > 0
    corr = float(np.corrcoef(pred, true)[0, 1]) if variable else 0.0
    if dataset_name == "simsv2":
        pred = np.clip(pred, -1.0, 1.0)
        true = np.clip(true, -1.0, 1.0)
        binary_pred = _bucketize(pred, [0.0])
        binary_true = _bucketize(true, [0.0])
        weak = np.abs(true) <= 0.4
        acc2_weak = (
            float(accuracy_score(binary_true[weak], binary_pred[weak])) if weak.any() else 0.0
        )
        acc3 = float(accuracy_score(_bucketize(true, [-0.1, 0.1]), _bucketize(pred, [-0.1, 0.1])))
        acc5 = float(
            accuracy_score(
                _bucketize(true, [-0.7, -0.1, 0.1, 0.7]),
                _bucketize(pred, [-0.7, -0.1, 0.1, 0.7]),
            )
        )
        return {
            "MAE": mae,
            "Corr": corr,
            "Acc2": float(accuracy_score(binary_true, binary_pred)),
            "F1": float(f1_score(binary_true, binary_pred, average="weighted", zero_division=0)),
            "Acc2Weak": acc2_weak,
            "Acc3": acc3,
            "Acc5": acc5,
            "Coverage": coverage,
        }

    binary_pred = pred >= 0
    binary_true = true >= 0
    acc2 = float(accuracy_score(binary_true, binary_pred))
    binary_f1 = float(f1_score(binary_true, binary_pred, average="weighted", zero_division=0))
    nonzero = true != 0
    nonzero_acc2 = (
        float(accuracy_score(true[nonzero] > 0, pred[nonzero] > 0)) if nonzero.any() else 0.0
    )
    nonzero_f1 = (
        float(f1_score(true[nonzero] > 0, pred[nonzero] > 0, average="weighted", zero_division=0))
        if nonzero.any()
        else 0.0
    )
    rounded_pred = np.clip(np.rint(pred), -3, 3)
    rounded_true = np.clip(np.rint(true), -3, 3)
    acc7 = float(accuracy_score(rounded_true, rounded_pred))
    return {
        "MAE": mae,
        "Corr": corr,
        "Acc2": acc2,
        "F1": binary_f1,
        "Acc7": acc7,
        "Non0Acc2": nonzero_acc2,
        "Non0F1": nonzero_f1,
        "Coverage": coverage,
    }


def classification_metrics(predictions: Iterable[int], targets: Iterable[int]) -> dict[str, float]:
    pred = np.asarray(list(predictions), dtype=np.int64)
    true = np.asarray(list(targets), dtype=np.int64)
    valid = pred >= 0
    return {
        "Acc": float(accuracy_score(true, pred)),
        "WF1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "Coverage": float(valid.mean()) if valid.size else 0.0,
    }


def compute_metrics(
    predictions: Iterable[float | int],
    targets: Iterable[float | int],
    config: DatasetConfig,
) -> dict[str, float]:
    if config.task == "regression":
        return regression_metrics(predictions, targets, config.name)
    if config.task == "classification":
        return classification_metrics(predictions, targets)
    raise ValueError(f"Unsupported task: {config.task}")
