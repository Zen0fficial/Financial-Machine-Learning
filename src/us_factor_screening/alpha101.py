"""Kakushadze (2016) 101 Formulaic Alphas - baseline OHLCV-only subset.

Each alpha is a function taking ``(open, high, low, close, volume, returns)``
panels (Date x Symbol DataFrames) and returning a Date x Symbol DataFrame.
The operators are imported from :mod:`alpha_operators` and combined to mirror
the original WQ formula text exactly.

Reference: Kakushadze, Z. (2016). 101 Formulaic Alphas. arXiv:1601.00991.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .alpha_operators import (
    abs,
    delay,
    delta,
    divide,
    eq,
    greater,
    if_cond,
    less,
    log,
    multiply,
    rank,
    sign,
    signed_power,
    subtract,
    ts_arg_max,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std_dev,
    ts_sum,
)

__all__ = [
    "ALPHA_101",
    "compute_alpha_101",
    "vwap_proxy",
    "alpha_001",
    "alpha_002",
    "alpha_003",
    "alpha_006",
    "alpha_007",
    "alpha_008",
    "alpha_009",
    "alpha_010",
    "alpha_012",
    "alpha_013",
    "alpha_015",
    "alpha_016",
    "alpha_017",
    "alpha_018",
    "alpha_019",
    "alpha_020",
    "alpha_035",
    "alpha_038",
    "alpha_040",
    "alpha_041",
    "alpha_044",
]


AlphaFunction = Callable[
    [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    pd.DataFrame,
]


def vwap_proxy(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame
) -> pd.DataFrame:
    """Typical-price VWAP proxy ``(high + low + close) / 3``.

    The free dataset has no intraday VWAP; the WorldQuant alpha definitions
    that reference ``vwap`` use this proxy as documented in the project README.
    """
    return (high + low + close) / 3.0


# ---------------------------------------------------------------------------
# Alpha implementations
# ---------------------------------------------------------------------------


def alpha_001(
    open: pd.DataFrame,  # noqa: A002 - mirrors WQ argument names
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#1: (rank(Ts_ArgMax(SignedPower((returns < 0 ? stddev(returns, 20) : close), 2.), 5)) - 0.5)."""
    inner = ts_std_dev(returns, 20).where(returns < 0, close)
    return rank(ts_arg_max(signed_power(inner, 2.0), 5)) - 0.5


def alpha_002(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))."""
    return -1.0 * ts_corr(
        rank(delta(log(volume), 2)),
        rank(divide(subtract(close, open), open)),
        6,
    )


def alpha_003(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))."""
    return -1.0 * ts_corr(rank(open), rank(volume), 10)


def alpha_006(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#6: (-1 * correlation(open, volume, 10))."""
    return -1.0 * ts_corr(open, volume, 10)


def alpha_007(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(correlation(vwap, close, 4), 0))) : (-1))."""
    vwap = vwap_proxy(high, low, close)
    adv20 = ts_mean(volume, 20)
    cond = less(adv20, volume)
    inner = multiply(
        -1.0 * ts_rank(abs(delta(close, 7)), 60),
        sign(delta(ts_corr(vwap, close, 4), 0)),
    )
    return if_cond(cond, inner, -1.0)


def alpha_008(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#8: (-1 * (rank((ts_sum(open, 5) * ts_sum(returns, 5) - delay((ts_sum(open, 5) * ts_sum(returns, 5)), 10)))))."""
    inner = multiply(ts_sum(open, 5), ts_sum(returns, 5))
    return -1.0 * rank(subtract(inner, delay(inner, 10)))


def alpha_009(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))."""
    d1 = delta(close, 1)
    cond1 = greater(ts_min(d1, 5), 0.0)
    cond2 = less(ts_max(d1, 5), 0.0)
    inner = if_cond(cond2, d1, -1.0 * d1)
    return if_cond(cond1, d1, inner)


def alpha_010(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))."""
    d1 = delta(close, 1)
    cond1 = greater(ts_min(d1, 4), 0.0)
    cond2 = less(ts_max(d1, 4), 0.0)
    inner = if_cond(cond2, d1, -1.0 * d1)
    return rank(if_cond(cond1, d1, inner))


def alpha_012(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))."""
    return multiply(sign(delta(volume, 1)), -1.0 * delta(close, 1))


def alpha_013(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))."""
    return -1.0 * rank(ts_cov(rank(close), rank(volume), 5))


def alpha_015(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#15: (-1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3))."""
    return -1.0 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3)


def alpha_016(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))."""
    return -1.0 * rank(ts_cov(rank(high), rank(volume), 5))


def alpha_017(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#17: ((((-1 * rank(ts_stddev(abs((close - open)), 5))) + correlation((close + open), volume, 10)) == -0.685923) ? -1 : (1 + large)).

    ``large`` is the WorldQuant sentinel for a very large positive number;
    we use ``1e9`` (finite but dominant) so the resulting panel remains finite.
    """
    large = 1e9
    inner = -1.0 * rank(ts_std_dev(abs(subtract(close, open)), 5)) + ts_corr(
        close + open, volume, 10
    )
    return if_cond(eq(inner, -0.685923), -1.0, 1.0 + large)


def alpha_018(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#18: (-1 * rank(((ts_stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))."""
    inner = (
        ts_std_dev(abs(subtract(close, open)), 5)
        + subtract(close, open)
        + ts_corr(close, open, 10)
    )
    return -1.0 * rank(inner)


def alpha_019(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + ts_sum(returns, 250)))))."""
    inner = subtract(close, delay(close, 7)) + delta(close, 7)
    return multiply(
        -1.0 * sign(inner),
        1.0 + rank(1.0 + ts_sum(returns, 250)),
    )


def alpha_020(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#20: ((((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1)))))."""
    return (
        -1.0 * rank(subtract(open, delay(high, 1)))
        * rank(subtract(open, delay(close, 1)))
        * rank(subtract(open, delay(low, 1)))
    )


def alpha_035(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#35: ((1 / rank((1 + ts_sum(returns, 16)))) * (1 - rank((1 + ts_sum(returns, 4)))))."""
    return multiply(
        divide(1.0, rank(1.0 + ts_sum(returns, 16))),
        1.0 - rank(1.0 + ts_sum(returns, 4)),
    )


def alpha_038(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#38: ((-1 * rank(ts_stddev(close, 10))) * correlation(close, volume, 10))."""
    return multiply(-1.0 * rank(ts_std_dev(close, 10)), ts_corr(close, volume, 10))


def alpha_040(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))."""
    return multiply(-1.0 * rank(ts_std_dev(high, 10)), ts_corr(high, volume, 10))


def alpha_041(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#41: (((high * low) ** 0.5) - vwap)."""
    vwap = vwap_proxy(high, low, close)
    return (high * low) ** 0.5 - vwap


def alpha_044(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Alpha#44: (-1 * correlation(high, ts_rank(volume, 5), 5))."""
    return -1.0 * ts_corr(high, ts_rank(volume, 5), 5)


# ---------------------------------------------------------------------------
# Registry and convenience runner
# ---------------------------------------------------------------------------


ALPHA_101: dict[str, AlphaFunction] = {
    "alpha_001": alpha_001,
    "alpha_002": alpha_002,
    "alpha_003": alpha_003,
    "alpha_006": alpha_006,
    "alpha_007": alpha_007,
    "alpha_008": alpha_008,
    "alpha_009": alpha_009,
    "alpha_010": alpha_010,
    "alpha_012": alpha_012,
    "alpha_013": alpha_013,
    "alpha_015": alpha_015,
    "alpha_016": alpha_016,
    "alpha_017": alpha_017,
    "alpha_018": alpha_018,
    "alpha_019": alpha_019,
    "alpha_020": alpha_020,
    "alpha_035": alpha_035,
    "alpha_038": alpha_038,
    "alpha_040": alpha_040,
    "alpha_041": alpha_041,
    "alpha_044": alpha_044,
}


def _returns_from_close(close: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close returns used when ``returns`` is not supplied."""
    return close.pct_change(fill_method=None)


def compute_alpha_101(
    frames: Mapping[str, pd.DataFrame],
    names: Iterable[str] | None = None,
    *,
    returns: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute selected alpha panels from an OHLCV frame mapping.

    ``frames`` must include ``open``, ``high``, ``low``, ``close`` and
    ``volume`` Date x Symbol DataFrames. ``returns`` defaults to the
    close-to-close return series.
    """
    required = ("open", "high", "low", "close", "volume")
    missing = [field for field in required if field not in frames]
    if missing:
        raise KeyError(f"frames is missing required OHLCV keys: {missing}")
    returns_frame = returns if returns is not None else _returns_from_close(frames["close"])
    selected = list(names) if names is not None else list(ALPHA_101)
    output: dict[str, pd.DataFrame] = {}
    for name in selected:
        if name not in ALPHA_101:
            raise KeyError(
                f"Unknown alpha {name!r}. Available: {', '.join(ALPHA_101)}"
            )
        values = ALPHA_101[name](
            frames["open"],
            frames["high"],
            frames["low"],
            frames["close"],
            frames["volume"],
            returns_frame,
        )
        output[name] = values.replace([np.inf, -np.inf], np.nan)
    return output
