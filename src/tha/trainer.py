from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RunConfig
from .metrics import compute_metrics
from .model import THAModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _scheduler(optimizer: AdamW, steps: int, warmup_ratio: float) -> LambdaLR:
    warmup = int(steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / warmup
        return max(0.0, float(steps - step) / max(steps - warmup, 1))

    return LambdaLR(optimizer, scale)


@torch.no_grad()
def evaluate(
    model: THAModel,
    loader: DataLoader[Any],
    config: RunConfig,
    prediction_path: str | Path | None = None,
) -> dict[str, float]:
    model.eval()
    predictions: list[float | int] = []
    targets: list[float | int] = []
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = _to_device(batch, model.device)
        output = model.generate(batch)
        batch_targets = batch["targets"].detach().cpu().tolist()
        predictions.extend(output["predictions"])
        targets.extend(batch_targets)
        rows.extend(
            {
                "id": sample_id,
                "target": target,
                "prediction": prediction,
                "decoded": decoded,
            }
            for sample_id, target, prediction, decoded in zip(
                batch["ids"], batch_targets, output["predictions"], output["texts"]
            )
        )
    if prediction_path is not None:
        destination = Path(prediction_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    return compute_metrics(predictions, targets, config.dataset)


def fit(
    model: THAModel,
    loaders: Mapping[str, DataLoader[Any]],
    config: RunConfig,
) -> dict[str, Any]:
    seed_everything(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.export_run_config(output_dir / "run_config.json")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loaders["train"]) / config.training.gradient_accumulation)
    scheduler = _scheduler(
        optimizer,
        updates_per_epoch * config.training.epochs,
        config.training.warmup_ratio,
    )
    maximize = config.dataset.metric_mode == "max"
    best = -float("inf") if maximize else float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(tqdm(loaders["train"], desc=f"epoch {epoch}"), start=1):
            batch = _to_device(batch, model.device)
            output = model(batch)
            loss = output["loss"] / config.training.gradient_accumulation
            loss.backward()
            losses.append(float(loss.detach()) * config.training.gradient_accumulation)
            if step % config.training.gradient_accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate(model, loaders["valid"], config)
        objective = metrics[config.dataset.primary_metric]
        improved = objective > best if maximize else objective < best
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if improved:
            best = objective
            stale_epochs = 0
            model.save_adapter(output_dir / "best_adapter.pt", {"epoch": epoch, "metrics": metrics})
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stop:
                break

    extra = model.load_adapter(output_dir / "best_adapter.pt")
    test_metrics = evaluate(
        model,
        loaders["test"],
        config,
        output_dir / "test_predictions.jsonl",
    )
    summary = {"best": extra, "test": test_metrics, "history": history}
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
