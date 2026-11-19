from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import compose_config
from .data import load_splits, make_loaders, print_description
from .model import THAModel
from .modules import ThalamusAdapter
from .trainer import evaluate, fit


def _common(parser: argparse.ArgumentParser, require_output: bool = False) -> None:
    parser.add_argument("--backbone", choices=("qwen18", "chatglm3", "llama2"), required=True)
    parser.add_argument("--dataset", choices=("mosei", "simsv2", "cherma"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=require_output, default="outputs")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the default training seed.",
    )
    parser.add_argument("--config-dir")


def _model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backbone-path", help="Optional local path in place of the configured model ID."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable-module",
        action="append",
        default=[],
        choices=sorted(ThalamusAdapter.MODULE_NAMES),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tha", description="THA reference implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-data", help="Validate files and print split statistics.")
    _common(inspect)

    train = commands.add_parser("train", help="Train the adapter while keeping the LLM frozen.")
    _common(train, require_output=True)
    _model_args(train)

    test = commands.add_parser("evaluate", help="Evaluate an adapter checkpoint.")
    _common(test, require_output=True)
    _model_args(test)
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--split", choices=("valid", "test"), default="test")
    return parser.parse_args()


def _config(args: argparse.Namespace):
    return compose_config(
        args.backbone,
        args.dataset,
        args.data_root,
        args.output_dir,
        args.seed,
        args.config_dir,
    )


def _build_model(args: argparse.Namespace, config):
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Pass --device cpu for diagnostics."
        )
    return THAModel.from_pretrained(
        config,
        args.backbone_path,
        args.device,
        args.disable_module,
    )


def main() -> int:
    args = parse_args()
    config = _config(args)
    if args.command == "inspect-data":
        print_description(load_splits(config.dataset))
        return 0

    model = _build_model(args, config)
    print(json.dumps({"trainable_parameters": model.trainable_parameters()}, indent=2))
    if args.command == "train":
        datasets = load_splits(config.dataset)
        loaders = make_loaders(datasets, config.training.batch_size, config.training.num_workers)
        fit(model, loaders, config)
        return 0

    model.load_adapter(args.checkpoint)
    dataset = load_splits(config.dataset, (args.split,))
    loader = make_loaders(dataset, config.training.batch_size, config.training.num_workers)[
        args.split
    ]
    destination = Path(config.output_dir) / f"{args.split}_predictions.jsonl"
    metrics = evaluate(model, loader, config, destination)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
