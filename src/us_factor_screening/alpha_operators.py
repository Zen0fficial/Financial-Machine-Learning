"""WorldQuant/Kakushadze operator vocabulary for panel factors.

All operators take and return ``pd.DataFrame`` panels indexed by date with
symbols in the columns (Date x Symbol). NaN values are propagated gracefully:
rolling operators only emit values once a full window of observations is
available, and arithmetic operators sanitise +/- inf to NaN so downstream
factor mining never sees non-finite numbers.

Reference: Kakushadze, Z. (2016). 101 Formulaic Alphas. arXiv:1601.00991.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

__all__ = [
    # Cross-sectional
    "rank",
    "scale",
    "zscore",
    # Time-series
    "delay",
    "delta",
    "ts_mean",
    "ts_sum",
    "ts_std_dev",
    "ts_var",
    "ts_min",
    "ts_max",
    "ts_rank",
    "ts_arg_min",
    "ts_arg_max",
    "ts_corr",
    "ts_cov",
    "decay_linear",
    "ts_skew",
    "ts_kurt",
    "ts_quantile",
    "ts_regression",
    "ts_regression_intercept",
    "ts_regression_beta",
    "ts_regression_residual",
    # Arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    "sign",
    "abs",
    "log",
    "power",
    "signed_power",
    "less",
    "greater",
    "eq",
    "max_",
    "min_",
    "if_cond",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace +/- inf with NaN to keep panels strictly finite-or-NaN."""
    return frame.replace([np.inf, -np.inf], np.nan)


def _rolling(frame: pd.DataFrame, n: int) -> pd.Rolling:
    """Rolling window that requires the full ``n`` observations."""
    if n < 1:
        raise ValueError("window length must be >= 1")
    return frame.rolling(n, min_periods=n)


# ---------------------------------------------------------------------------
# Cross-sectional operators (operate across stocks for each date)
# ---------------------------------------------------------------------------


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [0, 1] for each date row."""
    return df.rank(axis=1, pct=True)


def scale(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally scale so the row sum of absolute values equals 1."""
    denominator = df.abs().sum(axis=1)
    return _sanitize(df.div(denominator, axis=0))


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: ``(df - row_mean) / row_std`` per date."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return _sanitize(df.sub(mean, axis=0).div(std, axis=0))


# ---------------------------------------------------------------------------
# Time-series operators (operate across time for each stock column)
# ---------------------------------------------------------------------------


def delay(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Lagged panel: ``df.shift(n)``."""
    return df.shift(n)


def delta(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Difference: ``df - df.shift(n)``."""
    return df - df.shift(n)


def ts_mean(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling mean over ``n`` observations."""
    return _rolling(df, n).mean()


def ts_sum(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling sum over ``n`` observations."""
    return _rolling(df, n).sum()


def ts_std_dev(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling sample standard deviation over ``n`` observations."""
    return _rolling(df, n).std()


def ts_var(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling sample variance over ``n`` observations."""
    return _rolling(df, n).var()


def ts_min(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling minimum over ``n`` observations."""
    return _rolling(df, n).min()


def ts_max(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling maximum over ``n`` observations."""
    return _rolling(df, n).max()


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling percentile rank (0 to 1) of the current value within the window."""
    return _rolling(df, n).rank(pct=True)


def ts_arg_min(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Position (0-indexed) of the minimum value within the rolling window."""
    return _rolling(df, n).apply(lambda x: float(np.argmin(x)), raw=True)


def ts_arg_max(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Position (0-indexed) of the maximum value within the rolling window."""
    return _rolling(df, n).apply(lambda x: float(np.argmax(x)), raw=True)


def ts_corr(df1: pd.DataFrame, df2: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling Pearson correlation between two panels.

    Pandas emits ``+/- inf`` when either series has zero variance within the
    window (correlation is undefined); we coerce those to NaN so downstream
    factor mining never sees non-finite values.
    """
    return _sanitize(_rolling(df1, n).corr(df2))


def ts_cov(df1: pd.DataFrame, df2: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling sample covariance between two panels."""
    return _rolling(df1, n).cov(df2)


def decay_linear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Linearly-decayed moving average.

    Weights are ``(n-1, n-2, ..., 1, 0)`` applied most-recent-first and
    normalised by their sum, matching the WorldQuant decay_linear semantics.
    For ``n == 1`` the weighted average collapses to the input itself.
    """
    if n < 1:
        raise ValueError("window length must be >= 1")
    if n == 1:
        return df.copy()
    # x[0] is oldest, x[-1] is newest -> newest gets the largest weight.
    weights = np.arange(n, dtype=float)
    weights /= weights.sum()
    return _rolling(df, n).apply(lambda x: float(np.dot(x, weights)), raw=True)


def ts_skew(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling unbiased skewness over ``n`` observations."""
    return _rolling(df, n).skew()


def ts_kurt(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling unbiased kurtosis over ``n`` observations."""
    return _rolling(df, n).kurt()


def ts_quantile(df: pd.DataFrame, n: int, q: float) -> pd.DataFrame:
    """Rolling quantile ``q`` over ``n`` observations."""
    return _rolling(df, n).quantile(q)


def ts_regression(
    y: pd.DataFrame, x: pd.DataFrame, n: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling OLS regression of ``y`` on ``x`` over ``n`` observations.

    Returns a ``(intercept, beta, residual)`` tuple of panels. The residual
    is the in-sample residual of the current observation given the trailing
    regression coefficients (y_t - alpha_t - beta_t * x_t).
    """
    mean_x = ts_mean(x, n)
    mean_y = ts_mean(y, n)
    var_x = ts_var(x, n)
    cov_xy = ts_cov(x, y, n)
    beta = _sanitize(cov_xy / var_x)
    intercept = mean_y - beta * mean_x
    residual = _sanitize(y - intercept - beta * x)
    return intercept, beta, residual


def ts_regression_intercept(y: pd.DataFrame, x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling regression intercept (alpha) of ``y`` on ``x``."""
    return ts_regression(y, x, n)[0]


def ts_regression_beta(y: pd.DataFrame, x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling regression slope (beta) of ``y`` on ``x``."""
    return ts_regression(y, x, n)[1]


def ts_regression_residual(y: pd.DataFrame, x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling regression residual of ``y`` on ``x``."""
    return ts_regression(y, x, n)[2]


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


def add(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Element-wise addition."""
    return a + b


def subtract(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Element-wise subtraction."""
    return a - b


def multiply(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Element-wise multiplication."""
    return a * b


def divide(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Element-wise division with NaN where the denominator is zero."""
    return _sanitize(a / b)


def sign(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise sign: -1, 0, or 1."""
    return np.sign(df)


def abs(df: pd.DataFrame) -> pd.DataFrame:  # noqa: A001 - intentionally mirrors WQ
    """Element-wise absolute value.

    Named to match the WorldQuant operator vocabulary; the builtin ``abs``
    is intentionally shadowed within modules that import this name.
    """
    return df.abs()


def log(df: pd.DataFrame) -> pd.DataFrame:
    """Natural log; non-positive values become NaN."""
    return np.log(df.where(df > 0))


def power(df: pd.DataFrame, n: float) -> pd.DataFrame:
    """Element-wise ``df ** n``."""
    return df ** n


def signed_power(df: pd.DataFrame, n: float) -> pd.DataFrame:
    """``sign(df) * abs(df) ** n`` preserving the sign of the input."""
    return np.sign(df) * np.abs(df) ** n


def _valid_mask(
    a: pd.DataFrame | float, b: pd.DataFrame | float
) -> pd.Series | pd.DataFrame:
    """Boolean mask that is True where both operands are non-NaN."""
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        return a.notna() & b.notna()
    if isinstance(a, pd.DataFrame):
        return a.notna()
    if isinstance(b, pd.DataFrame):
        return b.notna()
    return True


def less(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Return 1.0 where ``a < b``, 0.0 where ``a >= b``, NaN where either is NaN."""
    comparison = (a < b).astype(float)
    mask = _valid_mask(a, b)
    if isinstance(mask, pd.DataFrame):
        return comparison.where(mask)
    return comparison


def greater(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Return 1.0 where ``a > b``, 0.0 where ``a <= b``, NaN where either is NaN."""
    comparison = (a > b).astype(float)
    mask = _valid_mask(a, b)
    if isinstance(mask, pd.DataFrame):
        return comparison.where(mask)
    return comparison


def eq(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Return 1.0 where ``a == b``, 0.0 otherwise, NaN where either is NaN."""
    comparison = (a == b).astype(float)
    mask = _valid_mask(a, b)
    if isinstance(mask, pd.DataFrame):
        return comparison.where(mask)
    return comparison


def max_(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Element-wise maximum (named to avoid shadowing the builtin)."""
    return np.maximum(a, b)


def min_(a: pd.DataFrame | float, b: pd.DataFrame | float) -> pd.DataFrame:
    """Element-wise minimum (named to avoid shadowing the builtin)."""
    return np.minimum(a, b)


def if_cond(
    cond: pd.DataFrame,
    a: pd.DataFrame | float,
    b: pd.DataFrame | float,
) -> pd.DataFrame:
    """Vectorial ``cond ? a : b`` returning ``a`` where ``cond > 0`` else ``b``.

    NaN in ``cond`` propagates to NaN in the result; NaN in the chosen branch
    is preserved.
    """
    cond_df = cond if isinstance(cond, pd.DataFrame) else pd.DataFrame(cond)
    a_df = (
        a
        if isinstance(a, pd.DataFrame)
        else pd.DataFrame(a, index=cond_df.index, columns=cond_df.columns)
    )
    b_df = (
        b
        if isinstance(b, pd.DataFrame)
        else pd.DataFrame(b, index=cond_df.index, columns=cond_df.columns)
    )
    chosen = a_df.where(cond_df > 0, b_df)
    return chosen.where(cond_df.notna())


# Convenience registry for downstream consumers (e.g. gplearn function sets).
ARITY: dict[str, tuple[int, Callable[..., object]]] = {
    # name: (arity, callable)
    "rank": (1, rank),
    "scale": (1, scale),
    "zscore": (1, zscore),
    "delay": (2, delay),
    "delta": (2, delta),
    "ts_mean": (2, ts_mean),
    "ts_sum": (2, ts_sum),
    "ts_std_dev": (2, ts_std_dev),
    "ts_var": (2, ts_var),
    "ts_min": (2, ts_min),
    "ts_max": (2, ts_max),
    "ts_rank": (2, ts_rank),
    "ts_arg_min": (2, ts_arg_min),
    "ts_arg_max": (2, ts_arg_max),
    "ts_corr": (3, ts_corr),
    "ts_cov": (3, ts_cov),
    "decay_linear": (2, decay_linear),
    "ts_skew": (2, ts_skew),
    "ts_kurt": (2, ts_kurt),
    "ts_quantile": (3, ts_quantile),
    "ts_regression_intercept": (3, ts_regression_intercept),
    "ts_regression_beta": (3, ts_regression_beta),
    "ts_regression_residual": (3, ts_regression_residual),
    "add": (2, add),
    "subtract": (2, subtract),
    "multiply": (2, multiply),
    "divide": (2, divide),
    "sign": (1, sign),
    "abs": (1, abs),
    "log": (1, log),
    "power": (2, power),
    "signed_power": (2, signed_power),
    "less": (2, less),
    "greater": (2, greater),
    "eq": (2, eq),
    "max_": (2, max_),
    "min_": (2, min_),
    "if_cond": (3, if_cond),
}
