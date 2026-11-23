from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gsq import SPLITS_DIR, cli


def _parse(*arguments: str):
    return cli.build_parser().parse_args(arguments)


@pytest.mark.parametrize("encoder", ["mlp", "gcn", "gesn"])
@pytest.mark.parametrize("pooler", ["mean", "deepsets", "gmnet"])
@pytest.mark.parametrize("predictor", ["softmax", "dirichlet"])
@pytest.mark.parametrize("scaler", ["none", "scalar", "classwise"])
def test_train_exposes_every_public_combination(
    monkeypatch, tmp_path: Path, encoder, pooler, predictor, scaler
):
    calls = []
    monkeypatch.setattr(cli, "_run", lambda module, overrides: calls.append((module, overrides)))
    args = _parse(
        "train",
        "--dataset", "cora",
        "--encoder", encoder,
        "--pooler", pooler,
        "--predictor", predictor,
        "--scaler", scaler,
        "--output", str(tmp_path / "run"),
        "--device", "cpu",
    )
    args.handler(args)
    module, overrides = calls.pop()
    assert module == "gsq.train"
    assert f"model/pooler={pooler}" in overrides
    assert f"model/head={predictor}" in overrides
    assert f"model.calibration_mode={scaler}" in overrides


@pytest.mark.parametrize("method", ["xnq", "damke"])
def test_baseline_is_a_single_explicit_run(monkeypatch, tmp_path: Path, method: str):
    calls = []
    monkeypatch.setattr(cli, "_run", lambda module, overrides: calls.append((module, overrides)))
    args = _parse(
        "train-baseline",
        "--dataset", "cora",
        "--method", method,
        "--protocol", "bfp",
        "--output", str(tmp_path / method),
        "--device", "cpu",
    )
    args.handler(args)
    module, overrides = calls.pop()
    assert module == "gsq.train"
    assert "model=aggregative" in overrides
    assert "model.calibration_mode=none" in overrides
    if method == "xnq":
        assert "model/encoder=gesn" in overrides
        assert "model/quantifier=sld" in overrides
    else:
        assert "model/encoder=appnp" in overrides
        assert "model/quantifier=kdey" in overrides
        assert "model.quantifier.structural_correction=true" in overrides


@pytest.mark.parametrize(
    ("directory", "nodes"),
    [
        ("amazon_ratings", 24_492),
        ("cora", 2_708),
        ("cora-2-vs-all", 2_708),
        ("toloker", 11_758),
    ],
)
def test_packaged_splits_are_disjoint_and_complete(directory: str, nodes: int):
    names = ("training", "calibration", "validation", "test")
    masks = [
        torch.load(SPLITS_DIR / directory / f"{name}_masks.pt", weights_only=True)
        for name in names
    ]
    assert all(mask.shape == (5, nodes) for mask in masks)
    for fold in range(5):
        membership = sum(mask[fold].to(torch.int8) for mask in masks)
        assert torch.all(membership == 1)


def test_amazon_uses_40_20_20_20_splits():
    root = SPLITS_DIR / "amazon_ratings"
    counts = {
        name: torch.load(root / f"{name}_masks.pt", weights_only=True).sum(dim=1)
        for name in ("training", "calibration", "validation", "test")
    }
    total = 24_492
    assert torch.all((counts["training"] / total - 0.4).abs() < 0.001)
    for name in ("calibration", "validation", "test"):
        assert torch.all((counts[name] / total - 0.2).abs() < 0.001)
