from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from math import log, sqrt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data import normalize_ohlcv

if TYPE_CHECKING:
    from .fundamentals import FundamentalsPanel

FactorFunction = Callable[["FactorContext"], pd.DataFrame]


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    description: str
    formula: str
    lookback: int
    required_fields: tuple[str, ...]
    direction: int
    compute: FactorFunction
    requires_benchmark: bool = False

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError("direction must be -1, 0, or 1")

    @property
    def direction_label(self) -> str:
        return {-1: "lower", 0: "unspecified", 1: "higher"}[self.direction]

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "formula": self.formula,
            "lookback": self.lookback,
            "required_fields": ",".join(self.required_fields),
            "requires_benchmark": self.requires_benchmark,
            "conventional_long_direction": self.direction_label,
        }


class FactorRegistry:
    def __init__(self, specs: Iterable[FactorSpec]) -> None:
        self._specs: dict[str, FactorSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"Duplicate factor name: {spec.name}")
            self._specs[spec.name] = spec

    def __iter__(self) -> Iterator[FactorSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, name: str) -> FactorSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            choices = ", ".join(self._specs)
            raise KeyError(f"Unknown factor {name!r}. Available factors: {choices}") from exc

    def names(self, family: str | None = None) -> list[str]:
        return [spec.name for spec in self if family is None or spec.family == family]

    def families(self) -> list[str]:
        return sorted({spec.family for spec in self})

    def manifest(self, names: Iterable[str] | None = None) -> pd.DataFrame:
        selected = list(names) if names is not None else self.names()
        return pd.DataFrame([self.get(name).as_record() for name in selected])


@dataclass(frozen=True)
class FactorContext:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    benchmark: str = "SPY"
    option_volume: pd.DataFrame | None = None
    fundamentals: FundamentalsPanel | None = None

    @classmethod
    def from_frames(
        cls,
        frames: dict[str, pd.DataFrame],
        benchmark: str = "SPY",
        option_volume_frames: dict[str, pd.DataFrame] | None = None,
        fundamentals: FundamentalsPanel | None = None,
    ) -> FactorContext:
        if not frames:
            raise ValueError("At least one OHLCV frame is required")
        normalized = {
            symbol.upper(): normalize_ohlcv(frame, symbol.upper())
            for symbol, frame in frames.items()
        }

        def field_matrix(field: str) -> pd.DataFrame:
            values = [
                frame.set_index("Date")[field].rename(symbol)
                for symbol, frame in normalized.items()
            ]
            return pd.concat(values, axis=1).sort_index().astype(float)

        close = field_matrix("Close")
        option_volume = _option_volume_matrix(option_volume_frames, close)
        return cls(
            open=field_matrix("Open"),
            high=field_matrix("High"),
            low=field_matrix("Low"),
            close=close,
            volume=field_matrix("Volume"),
            benchmark=benchmark.upper(),
            option_volume=option_volume,
            fundamentals=fundamentals,
        )

    @cached_property
    def returns(self) -> pd.DataFrame:
        return self.close.pct_change(fill_method=None)

    @cached_property
    def dollar_volume(self) -> pd.DataFrame:
        return self.close * self.volume

    @cached_property
    def market_returns(self) -> pd.Series:
        if self.benchmark not in self.returns.columns:
            raise ValueError(
                f"Benchmark {self.benchmark!r} is required for market-relative factors"
            )
        return self.returns[self.benchmark]


def _sanitize(values: pd.DataFrame) -> pd.DataFrame:
    return values.replace([np.inf, -np.inf], np.nan)


def _option_volume_matrix(
    option_volume_frames: dict[str, pd.DataFrame] | None,
    template: pd.DataFrame,
) -> pd.DataFrame | None:
    """Build a Date x Symbol option-volume matrix aligned to ``template``."""
    if option_volume_frames is None:
        return None
    series: list[pd.Series] = []
    for symbol, frame in option_volume_frames.items():
        data = frame.copy()
        if "Date" not in data.columns:
            data = data.reset_index()
            if "Date" not in data.columns:
                data = data.rename(columns={data.columns[0]: "Date"})
        data["Date"] = (
            pd.to_datetime(data["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        )
        data = data.dropna(subset=["Date"]).sort_values("Date")
        column = "OptionVolume" if "OptionVolume" in data.columns else data.columns[1]
        series.append(data.set_index("Date")[column].astype(float).rename(symbol.upper()))
    if not series:
        return None
    matrix = pd.concat(series, axis=1).sort_index()
    return matrix.reindex(index=template.index, columns=template.columns)


def _require_option_volume(context: FactorContext) -> pd.DataFrame:
    if context.option_volume is None:
        raise ValueError(
            "Option-volume factor requires context.option_volume; pass "
            "option_volume_frames to FactorContext.from_frames / compute_factor_zoo."
        )
    return context.option_volume


def _return(window: int) -> FactorFunction:
    return lambda context: context.close / context.close.shift(window) - 1.0


def _reversal(window: int) -> FactorFunction:
    return lambda context: -(context.close / context.close.shift(window) - 1.0)


def _momentum_skip(lookback: int, skip_recent: int) -> FactorFunction:
    return lambda context: context.close.shift(skip_recent) / context.close.shift(lookback) - 1.0


def _relative_momentum(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        asset = context.close / context.close.shift(window) - 1.0
        market = asset[context.benchmark]
        return asset.sub(market, axis=0)

    return compute


def _distance_to_sma(window: int) -> FactorFunction:
    return lambda context: (
        context.close / context.close.rolling(window, min_periods=window).mean() - 1.0
    )


def _drawdown(window: int) -> FactorFunction:
    return lambda context: (
        context.close / context.close.rolling(window, min_periods=window).max() - 1.0
    )


def _rsi(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        delta = context.close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        relative_strength = gain / loss
        return _sanitize(100.0 - 100.0 / (1.0 + relative_strength))

    return compute


def _realized_volatility(window: int) -> FactorFunction:
    return lambda context: context.returns.rolling(window, min_periods=window).std() * sqrt(252)


def _downside_volatility(window: int) -> FactorFunction:
    return lambda context: (
        context.returns.clip(upper=0).pow(2).rolling(window, min_periods=window).mean().pow(0.5)
        * sqrt(252)
    )


def _parkinson_volatility(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        log_range_squared = np.log(context.high / context.low).pow(2)
        variance = log_range_squared.rolling(window, min_periods=window).mean() * 252 / (4 * log(2))
        return _sanitize(variance.pow(0.5))

    return compute


def _market_covariance(context: FactorContext, window: int) -> pd.DataFrame:
    return context.returns.rolling(window, min_periods=window).cov(context.market_returns)


def _market_beta(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        covariance = _market_covariance(context, window)
        market_variance = context.market_returns.rolling(window, min_periods=window).var()
        return _sanitize(covariance.div(market_variance, axis=0))

    return compute


def _market_correlation(window: int) -> FactorFunction:
    return lambda context: context.returns.rolling(window, min_periods=window).corr(
        context.market_returns
    )


def _idiosyncratic_volatility(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        covariance = _market_covariance(context, window)
        market_variance = context.market_returns.rolling(window, min_periods=window).var()
        asset_variance = context.returns.rolling(window, min_periods=window).var()
        residual_variance = asset_variance.sub(covariance.pow(2).div(market_variance, axis=0))
        return _sanitize(residual_variance.clip(lower=0).pow(0.5) * sqrt(252))

    return compute


def _return_skew(window: int) -> FactorFunction:
    return lambda context: context.returns.rolling(window, min_periods=window).skew()


def _max_return(window: int) -> FactorFunction:
    return lambda context: context.returns.rolling(window, min_periods=window).max()


def _log_dollar_volume(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        average = context.dollar_volume.rolling(window, min_periods=window).mean()
        return _sanitize(np.log(average.where(average > 0)))

    return compute


def _amihud_illiquidity(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        daily = context.returns.abs() / context.dollar_volume.where(context.dollar_volume > 0)
        average = daily.rolling(window, min_periods=window).mean() * 1_000_000
        return _sanitize(np.log(average.where(average > 0)))

    return compute


def _volume_surprise(window: int) -> FactorFunction:
    return lambda context: (
        context.volume / context.volume.rolling(window, min_periods=window).median() - 1.0
    )


def _volume_trend(short_window: int, long_window: int) -> FactorFunction:
    return lambda context: (
        context.volume.rolling(short_window, min_periods=short_window).mean()
        / context.volume.rolling(long_window, min_periods=long_window).mean()
        - 1.0
    )


def _signed_volume_pressure(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        signed_volume = np.sign(context.returns) * context.volume
        numerator = signed_volume.rolling(window, min_periods=window).sum()
        denominator = context.volume.rolling(window, min_periods=window).sum()
        return _sanitize(numerator / denominator.where(denominator > 0))

    return compute


def _intraday_return(context: FactorContext) -> pd.DataFrame:
    return _sanitize(context.close / context.open - 1.0)


def _overnight_return(context: FactorContext) -> pd.DataFrame:
    return _sanitize(context.open / context.close.shift(1) - 1.0)


def _close_location(window: int) -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        spread = context.high - context.low
        location = (2 * context.close - context.high - context.low) / spread.where(spread > 0)
        return location.rolling(window, min_periods=window).mean()

    return compute


# ---------------------------------------------------------------------------
# Momentum family extensions
# ---------------------------------------------------------------------------


def _high_252d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        rolling_max = context.close.rolling(252, min_periods=63).max()
        return _sanitize(context.close / rolling_max.where(rolling_max > 0))

    return compute


def _ts_momentum_252d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        return _sanitize(np.sign(context.close / context.close.shift(252) - 1.0))

    return compute


def _vol_managed_momentum_126d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        ret = context.close / context.close.shift(126) - 1.0
        vol = context.returns.rolling(126, min_periods=63).std() * sqrt(252)
        return _sanitize(ret / vol.where(vol > 0))

    return compute


def _momentum_crash_protected_252d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        momentum = context.close.shift(21) / context.close.shift(252) - 1.0
        market_vol = context.market_returns.rolling(60, min_periods=60).std() * sqrt(252)
        scaler = np.exp(-np.maximum(0.0, market_vol - 0.12))
        return _sanitize(momentum.mul(scaler, axis=0))

    return compute


def _reversal_63d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        return _sanitize(-(context.close / context.close.shift(63) - 1.0))

    return compute


# ---------------------------------------------------------------------------
# Risk family extensions
# ---------------------------------------------------------------------------


def _downside_beta_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        returns = context.returns
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        r_vals = returns.to_numpy()
        rm_vals = context.market_returns.to_numpy()
        for i in range(60, len(returns)):
            r_block = r_vals[i - 60 : i]
            rm_block = rm_vals[i - 60 : i]
            down = rm_block < 0
            if int(down.sum()) < 10:
                continue
            rm_down = rm_block[down]
            r_down = r_block[down]
            rm_var = rm_down.var(ddof=1)
            if not np.isfinite(rm_var) or rm_var <= 0:
                continue
            rm_dm = rm_down - rm_down.mean()
            r_dm = r_down - np.nanmean(r_down, axis=0)
            cov = np.nanmean(r_dm * rm_dm[:, None], axis=0)
            result.iloc[i] = cov / rm_var
        return _sanitize(result)

    return compute


def _co_skew_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        returns = context.returns
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        r_vals = returns.to_numpy()
        rm_vals = context.market_returns.to_numpy()
        for i in range(60, len(returns)):
            r_block = r_vals[i - 60 : i]
            rm_block = rm_vals[i - 60 : i]
            rm_std = rm_block.std(ddof=1)
            if not np.isfinite(rm_std) or rm_std <= 0:
                continue
            rm_dm = rm_block - rm_block.mean()
            r_dm = r_block - np.nanmean(r_block, axis=0)
            coskew = np.nanmean(r_dm * (rm_dm[:, None] ** 2), axis=0)
            result.iloc[i] = coskew / (rm_std**3)
        return _sanitize(result)

    return compute


def _return_kurtosis(window: int) -> FactorFunction:
    return lambda context: context.returns.rolling(window, min_periods=window).kurt()


def _tail_var_252d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        return _sanitize(
            context.returns.rolling(252, min_periods=63).quantile(0.05)
        )

    return compute


def _co_kurtosis_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        returns = context.returns
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        r_vals = returns.to_numpy()
        rm_vals = context.market_returns.to_numpy()
        for i in range(60, len(returns)):
            r_block = r_vals[i - 60 : i]
            rm_block = rm_vals[i - 60 : i]
            rm_std = rm_block.std(ddof=1)
            if not np.isfinite(rm_std) or rm_std <= 0:
                continue
            rm_dm = rm_block - rm_block.mean()
            r_dm = r_block - np.nanmean(r_block, axis=0)
            cokurt = np.nanmean(r_dm * (rm_dm[:, None] ** 3), axis=0)
            result.iloc[i] = cokurt / (rm_std**4)
        return _sanitize(result)

    return compute


def _beta_bab_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        beta = _market_beta(60)(context)
        return _sanitize(0.6 * beta + 0.4)

    return compute


# ---------------------------------------------------------------------------
# Liquidity family extensions
# ---------------------------------------------------------------------------


def _roll_spread_20d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        price = context.close
        delta = price.diff()
        cov = delta.rolling(20, min_periods=20).cov(delta.shift(1))
        spread = 2.0 * np.sqrt((-cov).clip(lower=0.0))
        return _sanitize(spread.where(cov < 0))

    return compute


def _corwin_schultz_spread_2d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        log_hl = np.log(context.high / context.low)
        log_hl_sum = log_hl.rolling(2).sum()
        two_day_high = context.high.rolling(2).max()
        two_day_low = context.low.rolling(2).min()
        log_range = np.log(two_day_high / two_day_low)
        beta = (log_hl_sum - log_range**2).clip(lower=0.0)
        sqrt_beta = np.sqrt(beta)
        denominator = 3.0 - 2.0 * sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - sqrt_beta) / denominator
        spread = 2.0 * (np.exp(alpha) - 1.0) / (np.exp(alpha) + 1.0)
        return _sanitize(spread.rolling(20, min_periods=20).mean())

    return compute


def _zero_volume_days_63d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        return _sanitize((context.volume == 0).rolling(63, min_periods=63).sum())

    return compute


def _pastor_stambaugh_liquidity_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        # Pástor-Stambaugh reversal beta proxy: how strongly yesterday's
        # signed volume predicts today's return. Lagging the signed-volume
        # interaction keeps the measure strictly point-in-time.
        lagged_signed_volume = (np.sign(context.returns) * context.volume).shift(1)
        corr = context.returns.rolling(60, min_periods=60).corr(lagged_signed_volume)
        return _sanitize(corr)

    return compute


def _liquidity_commonality_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        dollar_volume = context.dollar_volume.where(context.dollar_volume > 0)
        amihud = context.returns.abs() / dollar_volume * 1_000_000
        market_amihud = amihud.mean(axis=1)
        corr = amihud.rolling(60, min_periods=60).corr(market_amihud)
        return _sanitize(corr)

    return compute


# ---------------------------------------------------------------------------
# Microstructure family extensions
# ---------------------------------------------------------------------------


def _price_delay_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        returns = context.returns
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        r_vals = returns.to_numpy()
        rm_vals = context.market_returns.to_numpy()
        n = len(returns)
        n_lags = 5
        rm_lagged = np.full((n, n_lags + 1), np.nan)
        for lag in range(n_lags + 1):
            rm_lagged[lag:, lag] = rm_vals[: n - lag] if lag else rm_vals
        ones = np.ones(60)
        for i in range(60, n):
            y = r_vals[i - 60 : i]
            x_full = rm_lagged[i - 60 : i]
            if np.isnan(y).any() or np.isnan(x_full).any():
                continue
            sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
            if (sst <= 0).any():
                continue
            x_concurrent = np.column_stack([ones, x_full[:, 0]])
            beta1, *_ = np.linalg.lstsq(x_concurrent, y, rcond=None)
            r2_1 = 1.0 - ((y - x_concurrent @ beta1) ** 2).sum(axis=0) / sst
            x_full_design = np.column_stack([ones, x_full])
            beta_full, *_ = np.linalg.lstsq(x_full_design, y, rcond=None)
            r2_full = 1.0 - ((y - x_full_design @ beta_full) ** 2).sum(axis=0) / sst
            with np.errstate(divide="ignore", invalid="ignore"):
                delay = 1.0 - r2_1 / np.where(r2_full > 1e-12, r2_full, np.nan)
            result.iloc[i] = delay
        return _sanitize(result)

    return compute


def _intraday_volatility_ratio_20d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        intraday = _intraday_return(context)
        overnight = _overnight_return(context)
        intraday_vol = intraday.rolling(20, min_periods=20).std()
        overnight_vol = overnight.rolling(20, min_periods=20).std()
        return _sanitize(intraday_vol / overnight_vol.where(overnight_vol > 0))

    return compute


# ---------------------------------------------------------------------------
# Trend / technical family extensions
# ---------------------------------------------------------------------------


def _macd_histogram_12_26_9() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        close = context.close
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        return _sanitize(macd - signal)

    return compute


def _bollinger_position_20d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        close = context.close
        sma = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std()
        return _sanitize((close - sma) / (2 * std.where(std > 0)))

    return compute


def _williams_r_14d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        highest = context.high.rolling(14, min_periods=14).max()
        lowest = context.low.rolling(14, min_periods=14).min()
        span = highest - lowest
        return _sanitize((highest - context.close) / span.where(span > 0))

    return compute


def _obv_slope_20d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        signed_volume = np.sign(context.returns) * context.volume
        obv = signed_volume.cumsum()
        t = np.arange(20, dtype=float)
        t_dm = t - t.mean()
        denom = float((t_dm**2).sum())

        def _slope(values: np.ndarray) -> float:
            if np.isnan(values).any():
                return np.nan
            y_dm = values - values.mean()
            return float((t_dm * y_dm).sum() / denom)

        return _sanitize(obv.rolling(20, min_periods=20).apply(_slope, raw=True))

    return compute


def _adx_14d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        high = context.high
        low = context.low
        prev_close = context.close.shift(1)
        true_range = np.maximum(
            np.maximum((high - low).to_numpy(), (high - prev_close).abs().to_numpy()),
            (low - prev_close).abs().to_numpy(),
        )
        true_range = pd.DataFrame(true_range, index=high.index, columns=high.columns)
        up_move = (high - high.shift(1)).clip(lower=0.0)
        down_move = (low.shift(1) - low).clip(lower=0.0)
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        period = 14
        alpha = 1.0 / period
        atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        plus_di = plus_di / atr.where(atr > 0)
        minus_di = minus_di / atr.where(atr > 0)
        di_sum = plus_di + minus_di
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum > 0)
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        return _sanitize(adx)

    return compute


# ---------------------------------------------------------------------------
# Option-volume family extensions
# ---------------------------------------------------------------------------


def _option_volume_trend_21_63d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        option_volume = _require_option_volume(context)
        short = option_volume.rolling(21, min_periods=21).mean()
        long = option_volume.rolling(63, min_periods=63).mean()
        return _sanitize(short / long.where(long > 0) - 1.0)

    return compute


def _option_volume_surprise_20d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        option_volume = _require_option_volume(context)
        mean = option_volume.rolling(20, min_periods=20).mean()
        std = option_volume.rolling(20, min_periods=20).std()
        return _sanitize((option_volume - mean) / std.where(std > 0))

    return compute


def _option_volume_ratio_20d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        option_volume = _require_option_volume(context)
        stock_volume = context.volume.where(context.volume > 0)
        ratio = option_volume / stock_volume
        return _sanitize(ratio.rolling(20, min_periods=20).mean())

    return compute


def compute_venue_imbalance_20d(
    venue_matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Venue-level option-volume imbalance averaged over 20 sessions.

    ``venue_matrices`` maps a venue name to a Date x Symbol DataFrame of option
    volume. The imbalance is ``(max_venue - min_venue) / sum_venues`` per
    date/symbol, then smoothed with a 20-day rolling mean.
    """
    if not venue_matrices:
        raise ValueError("at least one venue matrix is required")
    frames = list(venue_matrices.values())
    stacked = np.stack([frame.to_numpy() for frame in frames], axis=0)
    vmax = np.nanmax(stacked, axis=0)
    vmin = np.nanmin(stacked, axis=0)
    vsum = np.nansum(stacked, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        imbalance = (vmax - vmin) / vsum
    template = frames[0]
    imbalance = np.where(vsum > 0, imbalance, np.nan)
    result = pd.DataFrame(imbalance, index=template.index, columns=template.columns)
    return _sanitize(result.rolling(20, min_periods=20).mean())


# ---------------------------------------------------------------------------
# Crowding family (cross-sectional)
# ---------------------------------------------------------------------------


def _rolling_returns_block(
    values: np.ndarray, window: int, i: int
) -> tuple[np.ndarray, np.ndarray] | None:
    block = values[i - window : i]
    valid_cols = ~np.isnan(block).any(axis=0)
    block = block[:, valid_cols]
    if block.shape[0] < window or block.shape[1] < 2:
        return None
    return block, valid_cols


def _absorption_ratio_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        returns = context.returns
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        values = returns.to_numpy()
        k = min(5, returns.shape[1])
        for i in range(60, len(returns)):
            block = _rolling_returns_block(values, 60, i)
            if block is None:
                continue
            data, _ = block
            corr = np.corrcoef(data, rowvar=False)
            if corr.shape[0] < 2:
                continue
            eigvals = np.linalg.eigvalsh(corr)
            eigvals = np.clip(eigvals, 0.0, None)
            total = eigvals.sum()
            if not np.isfinite(total) or total <= 0:
                continue
            ratio = np.sort(eigvals)[-k:].sum() / total
            result.iloc[i] = ratio
        return _sanitize(result)

    return compute


def _pairwise_correlation_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        returns = context.returns
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        values = returns.to_numpy()
        for i in range(60, len(returns)):
            block = _rolling_returns_block(values, 60, i)
            if block is None:
                continue
            data, _ = block
            corr = np.corrcoef(data, rowvar=False)
            if corr.shape[0] < 2:
                continue
            upper = corr[np.triu_indices(corr.shape[0], k=1)]
            if upper.size == 0:
                continue
            result.iloc[i] = np.nanmean(upper)
        return _sanitize(result)

    return compute


def _comomentum_60d() -> FactorFunction:
    def compute(context: FactorContext) -> pd.DataFrame:
        returns = context.returns
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        values = returns.to_numpy()
        for i in range(60, len(returns)):
            block = _rolling_returns_block(values, 60, i)
            if block is None:
                continue
            data, _ = block
            portfolio = data.mean(axis=1)
            portfolio_var = portfolio.var(ddof=1)
            if not np.isfinite(portfolio_var) or portfolio_var <= 0:
                continue
            individual_var = data.var(axis=0, ddof=1).mean()
            if not np.isfinite(individual_var) or individual_var <= 0:
                continue
            result.iloc[i] = portfolio_var / individual_var
        return _sanitize(result)

    return compute


def _spec(
    name: str,
    family: str,
    description: str,
    formula: str,
    lookback: int,
    fields: tuple[str, ...],
    direction: int,
    compute: FactorFunction,
    requires_benchmark: bool = False,
) -> FactorSpec:
    return FactorSpec(
        name=name,
        family=family,
        description=description,
        formula=formula,
        lookback=lookback,
        required_fields=fields,
        direction=direction,
        compute=compute,
        requires_benchmark=requires_benchmark,
    )


FACTOR_REGISTRY = FactorRegistry(
    [
        _spec(
            "reversal_1d",
            "reversal",
            "One-session reversal",
            "-(C/C[-1]-1)",
            1,
            ("Close",),
            1,
            _reversal(1),
        ),
        _spec(
            "reversal_5d",
            "reversal",
            "One-week reversal",
            "-(C/C[-5]-1)",
            5,
            ("Close",),
            1,
            _reversal(5),
        ),
        _spec(
            "reversal_21d",
            "reversal",
            "One-month reversal",
            "-(C/C[-21]-1)",
            21,
            ("Close",),
            1,
            _reversal(21),
        ),
        _spec(
            "momentum_21d",
            "momentum",
            "One-month price momentum",
            "C/C[-21]-1",
            21,
            ("Close",),
            1,
            _return(21),
        ),
        _spec(
            "momentum_63d",
            "momentum",
            "Three-month price momentum",
            "C/C[-63]-1",
            63,
            ("Close",),
            1,
            _return(63),
        ),
        _spec(
            "momentum_126d",
            "momentum",
            "Six-month price momentum",
            "C/C[-126]-1",
            126,
            ("Close",),
            1,
            _return(126),
        ),
        _spec(
            "momentum_252d",
            "momentum",
            "Twelve-month price momentum",
            "C/C[-252]-1",
            252,
            ("Close",),
            1,
            _return(252),
        ),
        _spec(
            "momentum_252_21d",
            "momentum",
            "Twelve-to-one-month momentum",
            "C[-21]/C[-252]-1",
            252,
            ("Close",),
            1,
            _momentum_skip(252, 21),
        ),
        _spec(
            "relative_momentum_63d",
            "momentum",
            "Three-month return relative to the benchmark",
            "R63-R63_mkt",
            63,
            ("Close",),
            1,
            _relative_momentum(63),
            requires_benchmark=True,
        ),
        _spec(
            "trend_sma_20d",
            "trend",
            "Distance above the 20-session moving average",
            "C/SMA20-1",
            20,
            ("Close",),
            1,
            _distance_to_sma(20),
        ),
        _spec(
            "trend_sma_50d",
            "trend",
            "Distance above the 50-session moving average",
            "C/SMA50-1",
            50,
            ("Close",),
            1,
            _distance_to_sma(50),
        ),
        _spec(
            "trend_sma_200d",
            "trend",
            "Distance above the 200-session moving average",
            "C/SMA200-1",
            200,
            ("Close",),
            1,
            _distance_to_sma(200),
        ),
        _spec(
            "drawdown_252d",
            "trend",
            "Distance from the trailing 252-session high",
            "C/MAX252(C)-1",
            252,
            ("Close",),
            1,
            _drawdown(252),
        ),
        _spec(
            "rsi_14d",
            "trend",
            "Wilder relative strength index",
            "100-100/(1+EWMA(gain)/EWMA(loss))",
            14,
            ("Close",),
            1,
            _rsi(14),
        ),
        _spec(
            "realized_vol_20d",
            "risk",
            "Annualized 20-session close-return volatility",
            "STD20(r)*sqrt(252)",
            20,
            ("Close",),
            -1,
            _realized_volatility(20),
        ),
        _spec(
            "realized_vol_60d",
            "risk",
            "Annualized 60-session close-return volatility",
            "STD60(r)*sqrt(252)",
            60,
            ("Close",),
            -1,
            _realized_volatility(60),
        ),
        _spec(
            "realized_vol_252d",
            "risk",
            "Annualized 252-session close-return volatility",
            "STD252(r)*sqrt(252)",
            252,
            ("Close",),
            -1,
            _realized_volatility(252),
        ),
        _spec(
            "downside_vol_60d",
            "risk",
            "Annualized downside semideviation",
            "SQRT(MEAN60(min(r,0)^2)*252)",
            60,
            ("Close",),
            -1,
            _downside_volatility(60),
        ),
        _spec(
            "parkinson_vol_20d",
            "risk",
            "Annualized high-low Parkinson volatility",
            "SQRT(252*MEAN20(log(H/L)^2)/(4log2))",
            20,
            ("High", "Low"),
            -1,
            _parkinson_volatility(20),
        ),
        _spec(
            "market_beta_60d",
            "risk",
            "Rolling beta to the benchmark",
            "COV60(r,r_mkt)/VAR60(r_mkt)",
            60,
            ("Close",),
            -1,
            _market_beta(60),
            requires_benchmark=True,
        ),
        _spec(
            "market_correlation_60d",
            "risk",
            "Rolling return correlation to the benchmark",
            "CORR60(r,r_mkt)",
            60,
            ("Close",),
            -1,
            _market_correlation(60),
            requires_benchmark=True,
        ),
        _spec(
            "idiosyncratic_vol_60d",
            "risk",
            "Annualized residual volatility versus the benchmark",
            "SQRT((VAR60(r)-COV60^2/VAR60_mkt)*252)",
            60,
            ("Close",),
            -1,
            _idiosyncratic_volatility(60),
            requires_benchmark=True,
        ),
        _spec(
            "return_skew_60d",
            "distribution",
            "Rolling daily-return skewness",
            "SKEW60(r)",
            60,
            ("Close",),
            -1,
            _return_skew(60),
        ),
        _spec(
            "max_return_20d",
            "distribution",
            "Largest daily return in 20 sessions",
            "MAX20(r)",
            20,
            ("Close",),
            -1,
            _max_return(20),
        ),
        _spec(
            "log_dollar_volume_20d",
            "liquidity",
            "Log average daily dollar volume",
            "LOG(MEAN20(C*V))",
            20,
            ("Close", "Volume"),
            0,
            _log_dollar_volume(20),
        ),
        _spec(
            "amihud_illiquidity_20d",
            "liquidity",
            "Log Amihud absolute-return price impact",
            "LOG(1e6*MEAN20(|r|/(C*V)))",
            20,
            ("Close", "Volume"),
            1,
            _amihud_illiquidity(20),
        ),
        _spec(
            "volume_surprise_20d",
            "liquidity",
            "Volume relative to its 20-session median",
            "V/MEDIAN20(V)-1",
            20,
            ("Volume",),
            0,
            _volume_surprise(20),
        ),
        _spec(
            "volume_trend_20_120d",
            "liquidity",
            "Short versus long average volume",
            "MEAN20(V)/MEAN120(V)-1",
            120,
            ("Volume",),
            1,
            _volume_trend(20, 120),
        ),
        _spec(
            "signed_volume_pressure_20d",
            "liquidity",
            "Return-signed volume pressure",
            "SUM20(SIGN(r)*V)/SUM20(V)",
            20,
            ("Close", "Volume"),
            1,
            _signed_volume_pressure(20),
        ),
        _spec(
            "intraday_return_1d",
            "microstructure",
            "Open-to-close return",
            "C/O-1",
            1,
            ("Open", "Close"),
            0,
            _intraday_return,
        ),
        _spec(
            "overnight_return_1d",
            "microstructure",
            "Prior-close-to-open return",
            "O/C[-1]-1",
            2,
            ("Open", "Close"),
            0,
            _overnight_return,
        ),
        _spec(
            "close_location_20d",
            "microstructure",
            "Average close location within the daily range",
            "MEAN20((2C-H-L)/(H-L))",
            20,
            ("High", "Low", "Close"),
            1,
            _close_location(20),
        ),
        _spec(
            "high_252d",
            "momentum",
            "Distance below the trailing 52-week high",
            "C/MAX252(C)",
            252,
            ("Close",),
            1,
            _high_252d(),
        ),
        _spec(
            "ts_momentum_252d",
            "momentum",
            "Time-series momentum sign over 12 months",
            "SIGN(R252)",
            252,
            ("Close",),
            1,
            _ts_momentum_252d(),
        ),
        _spec(
            "vol_managed_momentum_126d",
            "momentum",
            "Six-month return scaled by annualized volatility",
            "R126/STD126(R)*sqrt(252)",
            126,
            ("Close",),
            1,
            _vol_managed_momentum_126d(),
        ),
        _spec(
            "momentum_crash_protected_252d",
            "momentum",
            "12-to-1 momentum scaled down when market vol is elevated",
            "R252_21*exp(-max(0,STD60(r_mkt)-0.12))",
            252,
            ("Close",),
            1,
            _momentum_crash_protected_252d(),
            requires_benchmark=True,
        ),
        _spec(
            "reversal_63d",
            "momentum",
            "Three-month price reversal",
            "-(C/C[-63]-1)",
            63,
            ("Close",),
            1,
            _reversal_63d(),
        ),
        _spec(
            "downside_beta_60d",
            "risk",
            "Rolling beta to the benchmark on down-market days",
            "COV60(r,r_mkt|r_mkt<0)/VAR60(r_mkt|r_mkt<0)",
            60,
            ("Close",),
            -1,
            _downside_beta_60d(),
            requires_benchmark=True,
        ),
        _spec(
            "co_skew_60d",
            "risk",
            "Co-skewness with the market return",
            "E[(r-mu)(r_mkt-mu_mkt)^2]/sigma_mkt^3",
            60,
            ("Close",),
            -1,
            _co_skew_60d(),
            requires_benchmark=True,
        ),
        _spec(
            "realized_kurtosis_60d",
            "risk",
            "Rolling daily-return kurtosis",
            "KURT60(r)",
            60,
            ("Close",),
            -1,
            _return_kurtosis(60),
        ),
        _spec(
            "tail_var_252d",
            "risk",
            "Rolling 5% historical value-at-risk",
            "QUANTILE5_252(r)",
            252,
            ("Close",),
            -1,
            _tail_var_252d(),
        ),
        _spec(
            "co_kurtosis_60d",
            "risk",
            "Co-kurtosis with the market return",
            "E[(r-mu)(r_mkt-mu_mkt)^3]/sigma_mkt^4",
            60,
            ("Close",),
            -1,
            _co_kurtosis_60d(),
            requires_benchmark=True,
        ),
        _spec(
            "beta_bab_60d",
            "risk",
            "BAB-style beta shrunk toward one",
            "0.6*beta_60+0.4",
            60,
            ("Close",),
            -1,
            _beta_bab_60d(),
            requires_benchmark=True,
        ),
        _spec(
            "roll_spread_20d",
            "liquidity",
            "Roll implicit bid-ask spread from price-change autocovariance",
            "2*SQRT(-COV20(dP,dP[-1]))",
            20,
            ("Close",),
            1,
            _roll_spread_20d(),
        ),
        _spec(
            "corwin_schultz_spread_2d",
            "liquidity",
            "Corwin-Schultz effective spread from high-low ratios",
            "2*(exp(alpha)-1)/(exp(alpha)+1)",
            20,
            ("High", "Low"),
            1,
            _corwin_schultz_spread_2d(),
        ),
        _spec(
            "zero_volume_days_63d",
            "liquidity",
            "Count of zero-volume sessions over 63 days",
            "COUNT63(V==0)",
            63,
            ("Volume",),
            1,
            _zero_volume_days_63d(),
        ),
        _spec(
            "pastor_stambaugh_liquidity_60d",
            "liquidity",
            "Reversal-based rolling liquidity proxy",
            "CORR60(r[+1],SIGN(r)*V)",
            60,
            ("Close", "Volume"),
            -1,
            _pastor_stambaugh_liquidity_60d(),
        ),
        _spec(
            "liquidity_commonality_60d",
            "liquidity",
            "Commonality in liquidity via Amihud co-movement",
            "CORR60(ILLIQ,ILLIQ_mkt)",
            60,
            ("Close", "Volume"),
            -1,
            _liquidity_commonality_60d(),
        ),
        _spec(
            "price_delay_60d",
            "microstructure",
            "Hou-Moskowitz price-delay proxy from R-squared ratios",
            "1-R2(concurrent)/R2(concurrent+lags)",
            60,
            ("Close",),
            -1,
            _price_delay_60d(),
            requires_benchmark=True,
        ),
        _spec(
            "intraday_volatility_ratio_20d",
            "microstructure",
            "Intraday to overnight return volatility ratio",
            "STD20(intraday_r)/STD20(overnight_r)",
            20,
            ("Open", "High", "Low", "Close"),
            0,
            _intraday_volatility_ratio_20d(),
        ),
        _spec(
            "macd_histogram_12_26_9",
            "trend",
            "MACD histogram (12,26,9)",
            "EMA12-EMA26-EMA9(EMA12-EMA26)",
            26,
            ("Close",),
            1,
            _macd_histogram_12_26_9(),
        ),
        _spec(
            "bollinger_position_20d",
            "trend",
            "Position within 20-day Bollinger bands",
            "(C-SMA20)/(2*STD20)",
            20,
            ("Close",),
            0,
            _bollinger_position_20d(),
        ),
        _spec(
            "williams_r_14d",
            "trend",
            "Williams %R oscillator",
            "(HH14-C)/(HH14-LL14)",
            14,
            ("High", "Low", "Close"),
            1,
            _williams_r_14d(),
        ),
        _spec(
            "obv_slope_20d",
            "trend",
            "Slope of on-balance volume over 20 sessions",
            "SLOPE20(OBV)",
            20,
            ("Close", "Volume"),
            1,
            _obv_slope_20d(),
        ),
        _spec(
            "adx_14d",
            "trend",
            "Wilder average directional index",
            "EMA14(|+DI--DI|/(+DI+-DI))",
            14,
            ("High", "Low", "Close"),
            1,
            _adx_14d(),
        ),
        _spec(
            "option_volume_trend_21_63d",
            "option_volume",
            "Short versus long option-volume trend",
            "MEAN21(OptVol)/MEAN63(OptVol)-1",
            63,
            ("OptionVolume",),
            0,
            _option_volume_trend_21_63d(),
        ),
        _spec(
            "option_volume_surprise_20d",
            "option_volume",
            "Z-scored option volume surprise",
            "(OptVol-MEAN20)/STD20",
            20,
            ("OptionVolume",),
            0,
            _option_volume_surprise_20d(),
        ),
        _spec(
            "option_volume_ratio_20d",
            "option_volume",
            "Option-to-stock volume ratio",
            "MEAN20(OptVol/V)",
            20,
            ("OptionVolume", "Volume"),
            -1,
            _option_volume_ratio_20d(),
        ),
        _spec(
            "absorption_ratio_60d",
            "crowding",
            "Fraction of return variance absorbed by top principal components",
            "SUM(eigen1-5)/SUM(all eigen)",
            60,
            ("Close",),
            -1,
            _absorption_ratio_60d(),
        ),
        _spec(
            "pairwise_correlation_60d",
            "crowding",
            "Average pairwise return correlation",
            "MEAN(pairwise CORR60)",
            60,
            ("Close",),
            -1,
            _pairwise_correlation_60d(),
        ),
        _spec(
            "comomentum_60d",
            "crowding",
            "Co-momentum ratio of portfolio to individual variance",
            "VAR(portfolio)/MEAN(VAR_i)",
            60,
            ("Close",),
            -1,
            _comomentum_60d(),
        ),
    ]
)


def compute_factor_zoo(
    frames: dict[str, pd.DataFrame],
    names: Iterable[str] | None = None,
    *,
    benchmark: str = "SPY",
    oriented: bool = False,
    option_volume_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute selected registered factors as Date x Symbol matrices."""
    selected = list(names) if names is not None else FACTOR_REGISTRY.names()
    context = FactorContext.from_frames(
        frames, benchmark=benchmark, option_volume_frames=option_volume_frames
    )
    output: dict[str, pd.DataFrame] = {}
    for name in selected:
        spec = FACTOR_REGISTRY.get(name)
        values = _sanitize(spec.compute(context)).reindex_like(context.close)
        if oriented and spec.direction == -1:
            values = -values
        output[name] = values
    return output


def orient_factor(values: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """Orient a raw factor so larger values follow its conventional long side."""
    spec = FACTOR_REGISTRY.get(factor_name)
    return -values if spec.direction == -1 else values.copy()
