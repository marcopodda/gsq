import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from gsq.utils.callbacks import instantiate_callbacks
from gsq.utils.logs import get_logger
from gsq.utils.paths import get_output_dir
from gsq.utils.seeding import seed_if_configured

from .reservoir import reservoir_cache_path

log = get_logger(__name__)
torch.set_float32_matmul_precision("medium")


def train(cfg: DictConfig) -> None:
    """Train one explicitly configured model and calibrate its best checkpoint."""
    output_dir = get_output_dir()
    checkpoint_path = output_dir / "model.ckpt"
    if checkpoint_path.exists() and not cfg.overwrite:
        log.info("Run already exists at %s; skipping.", output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.overwrite:
        for name in ("model.ckpt", "metrics.csv", "hparams.yaml", "config.yaml"):
            (output_dir / name).unlink(missing_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    seed_if_configured(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.datamodule, loader=cfg.loader, _recursive_=False)
    datamodule.setup(stage="fit")
    data = datamodule.dataset[0]
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    cache_path = reservoir_cache_path(cfg)
    model.prepare_reservoir(
        data,
        str(cache_path) if cache_path is not None else None,
        require_cached=cache_path is not None,
    )
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=instantiate_callbacks(cfg.callbacks),
        logger=logger,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    best_path = trainer.checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError("Training did not produce a checkpoint.")
    state = torch.load(best_path, map_location=model.device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    data = data.to(model.device)
    model.prepare_reservoir(
        data,
        str(cache_path) if cache_path is not None else None,
        require_cached=cache_path is not None,
    )
    calibration_metrics = model.fit_calibrators(data)
    trainer.save_checkpoint(best_path)
    logger.log_metrics(calibration_metrics, step=trainer.global_step + 1)
    logger.save()
    log.info(
        "Completed %s; calibration=%s; checkpoint=%s",
        output_dir,
        model.selected_calibration,
        best_path,
    )
