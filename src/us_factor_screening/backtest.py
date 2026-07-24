from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import bt
import pandas as pd


@dataclass
class BacktestArtifacts:
    stats: pd.DataFrame
    equity_curve: pd.DataFrame
    target_weights: pd.DataFrame
    realized_weights: pd.DataFrame
    raw_result: object


def _commission(quantity: float, price: float, bps: float) -> float:
    return abs(quantity) * price * bps / 10_000.0


def run_weight_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    name: str = "factor_strategy",
    benchmark_symbol: str | None = "SPY",
    initial_capital: float = 1_000_000.0,
    commission_bps: float = 5.0,
) -> BacktestArtifacts:
    """Run target weights through bt's portfolio accounting and analyzers."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if commission_bps < 0:
        raise ValueError("commission_bps cannot be negative")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")

    clean_prices = prices.sort_index().astype(float).dropna(how="any")
    if clean_prices.empty:
        raise ValueError("prices have no complete rows")
    if (clean_prices <= 0).any().any():
        raise ValueError("prices must be positive")

    weights = target_weights.reindex(columns=clean_prices.columns).fillna(0.0)
    weights = weights.loc[weights.index.intersection(clean_prices.index)].sort_index()
    if weights.empty:
        raise ValueError("target_weights do not overlap the price index")
    if (weights.abs().sum(axis=1) > 1.0000001).any():
        raise ValueError("target gross exposure cannot exceed 1.0")

    strategy = bt.Strategy(
        name,
        [
            bt.algos.WeighTarget(weights),
            bt.algos.Rebalance(),
        ],
    )
    commission_fn = partial(_commission, bps=commission_bps)
    tests = [
        bt.Backtest(
            strategy,
            clean_prices,
            initial_capital=initial_capital,
            commissions=commission_fn,
            integer_positions=False,
        )
    ]

    if benchmark_symbol is not None:
        benchmark = benchmark_symbol.upper()
        if benchmark not in clean_prices.columns:
            raise ValueError(f"benchmark {benchmark!r} is not present in prices")
        benchmark_weights = pd.DataFrame(
            0.0,
            index=[clean_prices.index[0]],
            columns=clean_prices.columns,
        )
        benchmark_weights.loc[:, benchmark] = 1.0
        benchmark_strategy = bt.Strategy(
            f"buy_hold_{benchmark}",
            [bt.algos.WeighTarget(benchmark_weights), bt.algos.Rebalance()],
        )
        tests.append(
            bt.Backtest(
                benchmark_strategy,
                clean_prices,
                initial_capital=initial_capital,
                commissions=commission_fn,
                integer_positions=False,
            )
        )

    result = bt.run(*tests)
    realized = result.backtests[name].security_weights.copy()
    return BacktestArtifacts(
        stats=result.stats.copy(),
        equity_curve=result.prices.copy(),
        target_weights=weights,
        realized_weights=realized,
        raw_result=result,
    )
