from lightning.pytorch import seed_everything


def seed_if_configured(seed: int | None) -> bool:
    """Seed all supported generators, including when the configured seed is zero."""
    if seed is None:
        return False
    seed_everything(seed, workers=True, verbose=False)
    return True
