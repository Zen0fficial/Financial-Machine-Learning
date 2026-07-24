from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_factor_screening.factors import momentum, rank_target_weights


def _prices() -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=90)
    steps = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "EWY": 50 + steps * 0.05,
            "SPY": 100 + steps * 0.20,
            "QQQ": 100 + steps * 0.35,
            "IWM": 100 - steps * 0.05,
        },
        index=index,
    )


def test_rank_weights_are_lagged_and_fully_invested() -> None:
    prices = _prices()
    factor = momentum(prices, lookback=20)
    weights = rank_target_weights(factor, prices, top_n=2, frequency="monthly")
    invested = weights[weights.sum(axis=1) > 0]

    assert not invested.empty
    assert np.allclose(invested.sum(axis=1), 1.0)
    assert (invested["QQQ"] == 0.5).all()
    assert (invested["SPY"] == 0.5).all()


def test_same_close_execution_is_rejected() -> None:
    prices = _prices()
    with pytest.raises(ValueError, match="lookahead"):
        rank_target_weights(prices, prices, signal_lag=0)


def test_long_short_weights_are_market_neutral() -> None:
    prices = _prices()
    factor = momentum(prices, lookback=20)
    weights = rank_target_weights(
        factor,
        prices,
        top_n=1,
        frequency="monthly",
        long_short=True,
    )
    invested = weights[weights.abs().sum(axis=1) > 0]
    assert np.allclose(invested.sum(axis=1), 0.0)
    assert np.allclose(invested.abs().sum(axis=1), 1.0)
