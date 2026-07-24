from __future__ import annotations

import numpy as np
import pandas as pd

from us_factor_screening.backtest import run_weight_backtest
from us_factor_screening.factor_zoo import compute_factor_zoo
from us_factor_screening.factors import rank_target_weights


def test_bt_runs_factor_strategy_and_benchmark() -> None:
    index = pd.bdate_range("2022-01-03", periods=300)
    steps = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "EWY": 60 + 0.04 * steps,
            "SPY": 100 + 0.08 * steps,
            "QQQ": 100 + 0.12 * steps,
            "IWM": 90 - 0.02 * steps,
        },
        index=index,
    )
    frames = {
        symbol: pd.DataFrame(
            {
                "Date": index,
                "Open": values,
                "High": values * 1.01,
                "Low": values * 0.99,
                "Close": values,
                "Volume": 1_000_000,
            }
        )
        for symbol, values in prices.items()
    }
    scores = compute_factor_zoo(frames, ["momentum_63d"])["momentum_63d"]
    weights = rank_target_weights(scores, prices, top_n=2)

    artifacts = run_weight_backtest(
        prices,
        weights,
        name="momentum_63",
        benchmark_symbol="SPY",
        commission_bps=5,
    )

    assert {"momentum_63", "buy_hold_SPY"}.issubset(artifacts.equity_curve.columns)
    assert not artifacts.stats.empty
    assert not artifacts.realized_weights.empty
    assert artifacts.equity_curve.iloc[-1]["momentum_63"] > 100
