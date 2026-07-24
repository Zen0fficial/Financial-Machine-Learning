from __future__ import annotations

import pandas as pd


def momentum(prices: pd.DataFrame, lookback: int = 63, skip_recent: int = 0) -> pd.DataFrame:
    """Trailing close-to-close return known at each row's close."""
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if skip_recent < 0:
        raise ValueError("skip_recent cannot be negative")
    reference = prices.shift(skip_recent)
    return reference / reference.shift(lookback) - 1.0


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "monthly") -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if frequency == "daily":
        return index
    periods = {"weekly": "W-FRI", "monthly": "M", "quarterly": "Q"}
    if frequency not in periods:
        raise ValueError("frequency must be daily, weekly, monthly, or quarterly")
    dates = index.to_series().groupby(index.to_period(periods[frequency])).max()
    return pd.DatetimeIndex(dates.to_numpy())


def rank_target_weights(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_n: int = 2,
    frequency: str = "monthly",
    signal_lag: int = 1,
    long_short: bool = False,
    target_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Convert cross-sectional factor scores into dated target weights.

    The default one-row lag ensures a close-derived signal is not traded at the
    same close that created it. Weights are emitted only on rebalance dates;
    bt holds the prior portfolio between those dates.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if signal_lag < 1:
        raise ValueError("signal_lag must be at least 1 to prevent lookahead")
    if not factor.columns.equals(prices.columns):
        factor = factor.reindex(columns=prices.columns)

    scores = factor.reindex(prices.index).shift(signal_lag)
    dates = (
        rebalance_dates(prices.index, frequency)
        if target_dates is None
        else pd.DatetimeIndex(target_dates).intersection(prices.index).sort_values()
    )
    weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)

    for date in dates:
        row = scores.loc[date]
        eligible = row[row.notna() & prices.loc[date].notna()].sort_values(ascending=False)
        required = top_n * (2 if long_short else 1)
        if len(eligible) < required:
            continue

        longs = eligible.head(top_n).index
        if long_short:
            shorts = eligible.tail(top_n).index
            weights.loc[date, longs] = 0.5 / top_n
            weights.loc[date, shorts] = -0.5 / top_n
        else:
            weights.loc[date, longs] = 1.0 / top_n
    return weights
