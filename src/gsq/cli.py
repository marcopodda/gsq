"""User-facing single-experiment command line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASETS = ("amazon", "cora", "cora-binary-2", "toloker")
PROTOCOLS = ("app", "bfp", "rwp")


def _run(module: str, overrides: list[str]) -> None:
    command = [sys.executable, "-m", module, *overrides]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def _output_override(path: str) -> str:
    return f"hydra.run.dir={Path(path).expanduser().resolve()}"


def _dataset_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)


def _protocol_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", choices=PROTOCOLS, default="app")


def _gesn_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gesn-channels", type=int, default=1024)
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--recurrent-scale", type=float, default=10.0)


def _training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)


def _base_training_overrides(args: argparse.Namespace) -> list[str]:
    return [
        f"experiment={args.dataset}",
        f"protocol={args.protocol}",
        f"datamodule.dataset.fold_index={args.fold}",
        f"trainer={args.device}",
        f"trainer.max_epochs={args.epochs}",
        f"model.optimizer.lr={args.learning_rate}",
        _output_override(args.output),
    ]


def pretrain_gesn(args: argparse.Namespace) -> None:
    common = [
        f"experiment={args.dataset}",
        "model/encoder=gesn",
        f"model.encoder.hidden_channels={args.channels}",
        f"model.encoder.input_scale={args.input_scale}",
        f"model.encoder.recurrent_scale={args.recurrent_scale}",
        f"device={args.device}",
    ]
    _run("gsq.cache_reservoir", common)
    if args.projection_channels:
        widths = sorted(set(args.projection_channels))
        _run(
            "gsq.cache_reservoir",
            [
                f"experiment={args.dataset}",
                "model/encoder=gesn_pca",
                f"model.encoder.reservoir_channels={args.channels}",
                f"model.encoder.hidden_channels={max(widths)}",
                f"model.encoder.input_scale={args.input_scale}",
                f"model.encoder.recurrent_scale={args.recurrent_scale}",
                f"projection_channels={widths}",
                f"device={args.device}",
            ],
        )


def train(args: argparse.Namespace) -> None:
    overrides = _base_training_overrides(args)
    encoder = args.encoder
    if encoder == "gesn":
        output_channels = args.hidden_channels or args.gesn_channels
        encoder_config = "gesn" if output_channels == args.gesn_channels else "gesn_pca"
        overrides.extend(
            [
                f"model/encoder={encoder_config}",
                f"model.encoder.hidden_channels={output_channels}",
                f"model.encoder.input_scale={args.input_scale}",
                f"model.encoder.recurrent_scale={args.recurrent_scale}",
            ]
        )
        if encoder_config == "gesn_pca":
            overrides.append(f"model.encoder.reservoir_channels={args.gesn_channels}")
    else:
        overrides.extend(
            [
                f"model/encoder={encoder}",
                f"model.encoder.hidden_channels={args.hidden_channels or 64}",
                f"model.encoder.num_layers={args.layers}",
            ]
        )
    overrides.extend(
        [
            f"model/pooler={args.pooler}",
            f"model/head={args.predictor}",
            f"model.calibration_mode={args.scaler}",
        ]
    )
    _run("gsq.train", overrides)


def train_baseline(args: argparse.Namespace) -> None:
    overrides = _base_training_overrides(args)
    overrides.extend(["model=aggregative", "model/classifier=linear"])
    if args.method == "xnq":
        overrides.extend(
            [
                "model/encoder=gesn",
                "model/quantifier=sld",
                f"model.encoder.hidden_channels={args.gesn_channels}",
                f"model.encoder.input_scale={args.input_scale}",
                f"model.encoder.recurrent_scale={args.recurrent_scale}",
                "model.calibration_mode=none",
            ]
        )
    else:
        correction = str(args.protocol in {"bfp", "rwp"}).lower()
        overrides.extend(
            [
                "model/encoder=appnp",
                "model/quantifier=kdey",
                "model.calibration_mode=none",
                f"model.quantifier.structural_correction={correction}",
                f"model.quantifier.bandwidth={args.bandwidth}",
            ]
        )
    _run("gsq.train", overrides)


def evaluate(args: argparse.Namespace) -> None:
    _run(
        "gsq.evaluate",
        [
            f"run_dir={Path(args.run_dir).expanduser().resolve()}",
            f"protocol={args.protocol}",
            f"task={args.split}",
            f"seed={args.seed}",
            _output_override(args.output),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain-gesn")
    pretrain.add_argument("--dataset", choices=DATASETS, required=True)
    pretrain.add_argument("--channels", type=int, default=1024)
    pretrain.add_argument("--input-scale", type=float, default=0.5)
    pretrain.add_argument("--recurrent-scale", type=float, default=10.0)
    pretrain.add_argument("--projection-channels", type=int, nargs="*")
    pretrain.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    pretrain.set_defaults(handler=pretrain_gesn)

    train_parser = subparsers.add_parser("train")
    _dataset_options(train_parser)
    _protocol_option(train_parser)
    _training_options(train_parser)
    _gesn_options(train_parser)
    train_parser.add_argument("--encoder", choices=("mlp", "gcn", "gesn"), required=True)
    train_parser.add_argument("--pooler", choices=("mean", "deepsets", "gmnet"), required=True)
    train_parser.add_argument("--predictor", choices=("softmax", "dirichlet"), required=True)
    train_parser.add_argument("--scaler", choices=("none", "scalar", "classwise"), default="none")
    train_parser.add_argument("--hidden-channels", type=int)
    train_parser.add_argument("--layers", type=int, default=2)
    train_parser.set_defaults(handler=train)

    baseline = subparsers.add_parser("train-baseline")
    _dataset_options(baseline)
    _protocol_option(baseline)
    _training_options(baseline)
    _gesn_options(baseline)
    baseline.add_argument("--method", choices=("xnq", "damke"), required=True)
    baseline.add_argument("--bandwidth", type=float, default=0.1)
    baseline.set_defaults(handler=train_baseline)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--run-dir", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--protocol", choices=PROTOCOLS, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
