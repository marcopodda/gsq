from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def serialize_prevalences(report: pd.DataFrame) -> pd.DataFrame:
    """Expand prevalence arrays into stable numeric CSV columns."""
    serialized = report.drop(columns=["true-prev", "estim-prev"]).copy()
    true = np.stack(report["true-prev"].to_numpy())
    estimated = np.stack(report["estim-prev"].to_numpy())
    for class_index in range(true.shape[1]):
        serialized[f"true_prev_{class_index}"] = true[:, class_index]
        serialized[f"estim_prev_{class_index}"] = estimated[:, class_index]
        serialized[f"signed_error_{class_index}"] = (
            estimated[:, class_index] - true[:, class_index]
        )
    serialized.insert(0, "sample", np.arange(len(serialized)))
    return serialized


def summarize_report(report: pd.DataFrame) -> dict[str, Any]:
    """Summarize errors, uncertainty, bias, and prevalence coverage."""
    true = np.stack(report["true-prev"].to_numpy())
    estimated = np.stack(report["estim-prev"].to_numpy())
    summary: dict[str, Any] = {
        "num_samples": len(report),
        "sample_size": None,
        "metrics": {},
        "per_class": {},
        "prevalence_coverage": {},
    }
    for metric in report.columns.difference(["true-prev", "estim-prev"]):
        values = report[metric].to_numpy(dtype=float)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
            "sem": float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0,
        }
    for class_index in range(true.shape[1]):
        errors = estimated[:, class_index] - true[:, class_index]
        summary["per_class"][str(class_index)] = {
            "bias": float(errors.mean()),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
        }
        summary["prevalence_coverage"][str(class_index)] = {
            "min": float(true[:, class_index].min()),
            "max": float(true[:, class_index].max()),
            "mean": float(true[:, class_index].mean()),
            "std": float(true[:, class_index].std(ddof=1)) if len(true) > 1 else 0.0,
        }
    return summary
