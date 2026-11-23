# GSQ

This code runs graph node quantification experiments. Tested on Linux, requires [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

## Installation

```bash
git clone https://github.com/marcopodda/gsq.git
cd gsq
uv sync
```

Datasets are downloaded automatically on first use into `~/.cache/gsq/data`. Hydra configurations live in `configs/`, while the versioned split masks live in `data/splits/`. We support 4 datasets: Cora, Cora-2, Amazon, and Toloker. All use deterministic five-fold 40/20/20/20 training/calibration/validation/test partitions.

## Commands

Precompute a reusable GESN representation:

```bash
uv run gsq pretrain-gesn \
  --dataset cora \
  --channels 1024 \
  --input-scale 0.5 \
  --recurrent-scale 10 \
  --projection-channels 64 \
  --device auto
```

Instantiate the framework and train it:

```bash
uv run gsq train \
  --dataset cora \
  --fold 0 \
  --protocol app \
  --encoder gcn \
  --pooler deepsets \
  --predictor softmax \
  --scaler scalar \
  --hidden-channels 64 \
  --layers 2 \
  --output runs/cora-gcn-deepsets
```

The supported framework components are:

- encoders: `mlp`, `gcn`, `gesn`;
- poolers: `mean`, `deepsets`, `gmnet`;
- predictors: `softmax`, `dirichlet`;
- scalers: `none`, `scalar`, `classwise`.

For GESN, `--gesn-channels` identifies the cached source representation and `--hidden-channels` identifies the representation consumed by the pooler. If the dimensions differ, GSQ loads the corresponding precomputed PCA view.

Train an aggregative baseline:

```bash
uv run gsq train-baseline \
  --method xnq \
  --dataset cora \
  --fold 0 \
  --protocol app \
  --output runs/cora-xnq

uv run gsq train-baseline \
  --method damke \
  --dataset cora \
  --fold 0 \
  --protocol bfp \
  --output runs/cora-damke
```

`xnq` is GESN-linear-SLD. `damke` is APPNP-linear-KDEy, with the
paper's structural correction enabled for BFP and RWP and disabled for APP.

Evaluate an explicitly selected training run:

```bash
uv run gsq evaluate \
  --run-dir runs/cora-gcn-deepsets \
  --protocol app \
  --split test \
  --output runs/cora-gcn-deepsets-test
```

Evaluation writes the resolved configuration, summary metrics, QuaPy's native report, and a numeric per-bag report containing true and estimated prevalence.

## Reproducibility

Every training run stores its resolved configuration, checkpoint, learning curve, and random seed.

Run the validation suite with:

```bash
uv run pytest
uv run ruff check
uv run ty check
```
