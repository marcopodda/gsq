from __future__ import annotations

from pathlib import Path

import quapy as qp
from omegaconf import DictConfig, OmegaConf
from quapy.protocol import APP, UPP

from gsq.utils.paths import get_output_dir
from gsq.utils.seeding import seed_if_configured
from gsq.utils.yaml import dump_yaml

from .protocols import BFPProtocol, RWPProtocol
from .reporting import serialize_prevalences, summarize_report
from .utils import load_experiment_data
from .wrapper import make_graph_quantifier


def make_protocol(quantifier, cfg: DictConfig):
    collection = quantifier.lc
    common = {
        "sample_size": cfg.protocol.sample_size,
        "random_state": cfg.seed,
    }
    if cfg.protocol.name == "app" and collection.n_classes == 2:
        return APP(
            collection,
            repeats=cfg.protocol.repeats,
            smooth_limits_epsilon=0.001,
            **common,
        )
    if cfg.protocol.name == "app":
        return UPP(collection, repeats=cfg.protocol.repeats, **common)
    structural = {**common, "edge_index": quantifier.edge_index}
    if cfg.protocol.name == "bfp":
        return BFPProtocol(
            collection,
            seeds_per_class=cfg.protocol.seeds_per_class,
            max_depth=cfg.protocol.max_depth,
            full_edge_index=quantifier.full_edge_index.numpy(),
            evaluation_nodes=quantifier.evaluation_indices.numpy(),
            **structural,
        )
    if cfg.protocol.name == "rwp":
        return RWPProtocol(
            collection,
            seeds_per_class=None,
            num_seeds=cfg.protocol.num_seeds,
            restart_probability=cfg.protocol.restart_probability,
            max_steps=cfg.protocol.max_steps,
            full_edge_index=quantifier.full_edge_index.numpy(),
            evaluation_nodes=quantifier.evaluation_indices.numpy(),
            **structural,
        )
    raise ValueError(f"Unsupported protocol: {cfg.protocol.name}")


def evaluate(cfg: DictConfig) -> None:
    """Evaluate one explicitly supplied training run."""
    if cfg.task not in {"test", "val"}:
        raise ValueError("task must be either 'test' or 'val'.")
    seed_if_configured(cfg.seed)
    qp.environ["SAMPLE_SIZE"] = cfg.protocol.sample_size

    run_dir = Path(cfg.run_dir).expanduser().resolve()
    loader, model, train_cfg = load_experiment_data(run_dir)
    quantifier = make_graph_quantifier(
        model,
        loader,
        mode=cfg.task,
        protocol_name=cfg.protocol.name,
    )
    protocol = make_protocol(quantifier, cfg)
    report = qp.evaluation.evaluation_report(
        quantifier,
        protocol,
        error_metrics=list(cfg.error_metrics),
        aggr_speedup=False,
    )
    if isinstance(protocol, (BFPProtocol, RWPProtocol)):
        report["sample-size"] = protocol.sample_sizes
    if isinstance(protocol, RWPProtocol):
        report["traversal-steps"] = protocol.traversal_steps

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_pickle(output_dir / "quapy_report.pkl")
    report.to_csv(output_dir / "quapy_report.csv", index=False)
    serialize_prevalences(report).to_csv(output_dir / "report.csv", index=False)
    summary = summarize_report(report)
    summary.update(
        {
            "source_run": str(run_dir),
            "dataset": train_cfg.name,
            "fold": int(train_cfg.datamodule.dataset.fold_index),
            "protocol": cfg.protocol.name,
            "split": cfg.task,
            "seed": int(cfg.seed),
            "timing": quantifier.timing(),
        }
    )
    dump_yaml(output_dir / "results.yaml", summary)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    metrics = summary["metrics"]
    print(
        f"AE={metrics['ae']['mean']:.6f} "
        f"RAE={metrics['rae']['mean']:.6f} "
        f"report={output_dir / 'report.csv'}"
    )
