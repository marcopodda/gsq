from unittest.mock import patch

import pytest

from gsq.utils.seeding import seed_if_configured


@pytest.mark.parametrize("seed", [0, 42])
def test_seed_if_configured_accepts_zero_and_positive_seeds(seed):
    with patch("gsq.utils.seeding.seed_everything") as seed_everything:
        assert seed_if_configured(seed)

    seed_everything.assert_called_once_with(seed, workers=True, verbose=False)


def test_seed_if_configured_skips_none():
    with patch("gsq.utils.seeding.seed_everything") as seed_everything:
        assert not seed_if_configured(None)

    seed_everything.assert_not_called()
