from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_factor_screening.factor_zoo import (
    FACTOR_REGISTRY,
    compute_factor_zoo,
    compute_venue_imbalance_20d,
    orient_factor,
)
from us_factor_screening.reporting import write_factor_outputs


def _frames(periods: int = 320) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    steps = np.arange(periods, dtype=float)
    output = {}
    for position, (symbol, slope) in enumerate(
        {"EWY": 0.03, "SPY": 0.05, "QQQ": 0.08, "IWM": -0.01}.items()
    ):
        close = 100 + position * 10 + slope * steps + np.sin(steps / 11 + position)
        open_price = close * (1 + 0.001 * np.cos(steps / 7 + position))
        high = np.maximum(open_price, close) * 1.01
        low = np.minimum(open_price, close) * 0.99
        output[symbol] = pd.DataFrame(
            {
                "Date": dates,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": 1_000_000 + position * 100_000 + steps * 100,
            }
        )
    return output


_SYMBOLS = ["EWY", "SPY", "QQQ", "IWM"]


def _option_volume_frames(periods: int = 320) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    steps = np.arange(periods, dtype=float)
    output: dict[str, pd.DataFrame] = {}
    for position, symbol in enumerate(_SYMBOLS):
        option_volume = (
            50_000
            + position * 10_000
            + steps * 50
            + np.sin(steps / 9 + position) * 5_000
        ).clip(min=1.0)
        output[symbol] = pd.DataFrame({"Date": dates, "OptionVolume": option_volume})
    return output


def _noisy_frames(periods: int = 360, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Realistic OHLCV panel with mean-reverting returns and an SPY benchmark."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=periods)
    steps = np.arange(periods, dtype=float)
    profiles = {
        "EWY": 0.0004,
        "SPY": 0.0006,
        "QQQ": 0.0009,
        "IWM": -0.0001,
        "AAPL": 0.0007,
        "MSFT": 0.0008,
        "NVDA": 0.0011,
        "GOOG": 0.0005,
    }
    output: dict[str, pd.DataFrame] = {}
    for position, (symbol, drift) in enumerate(profiles.items()):
        shocks = rng.normal(0.0, 0.012, periods)
        ar = np.empty(periods)
        ar[0] = shocks[0]
        for t in range(1, periods):
            ar[t] = -0.25 * ar[t - 1] + shocks[t]
        log_returns = drift + ar + np.sin(steps / 11 + position) * 0.001
        close = np.empty(periods)
        price = 100.0 + position * 10
        for t in range(periods):
            close[t] = price
            price *= 1.0 + log_returns[t]
        open_price = close * (1.0 + rng.normal(0.0, 0.002, periods))
        high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.0, 0.006, periods))
        low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.0, 0.006, periods))
        volume = (
            1_000_000
            + position * 100_000
            + rng.integers(0, 400_000, periods)
        ).astype(float)
        output[symbol] = pd.DataFrame(
            {
                "Date": dates,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
    return output


def test_registry_has_unique_documented_factors() -> None:
    names = FACTOR_REGISTRY.names()
    assert len(names) == len(set(names))
    assert len(names) >= 30
    assert {
        "momentum",
        "reversal",
        "trend",
        "risk",
        "distribution",
        "liquidity",
        "microstructure",
        "option_volume",
        "crowding",
    } == set(FACTOR_REGISTRY.families())
    manifest = FACTOR_REGISTRY.manifest()
    assert manifest.notna().all().all()
    benchmark_factors = set(manifest.loc[manifest["requires_benchmark"], "name"])
    assert benchmark_factors == {
        "relative_momentum_63d",
        "market_beta_60d",
        "market_correlation_60d",
        "idiosyncratic_vol_60d",
        "momentum_crash_protected_252d",
        "downside_beta_60d",
        "co_skew_60d",
        "co_kurtosis_60d",
        "beta_bab_60d",
        "price_delay_60d",
    }


def test_all_factors_compute_with_expected_shape() -> None:
    frames = _frames()
    factors = compute_factor_zoo(
        frames, option_volume_frames=_option_volume_frames()
    )
    expected_index = pd.DatetimeIndex(frames["SPY"]["Date"])

    assert set(factors) == set(FACTOR_REGISTRY.names())
    for values in factors.values():
        assert values.index.equals(expected_index)
        assert list(values.columns) == ["EWY", "SPY", "QQQ", "IWM"]
        assert np.isfinite(values.to_numpy()[~values.isna().to_numpy()]).all()


def test_known_momentum_and_reversal_formulas() -> None:
    frames = _frames(30)
    factors = compute_factor_zoo(frames, ["momentum_21d", "reversal_21d"])
    close = frames["EWY"].set_index("Date")["Close"]
    expected = close.iloc[-1] / close.iloc[-22] - 1

    assert factors["momentum_21d"].iloc[-1]["EWY"] == pytest.approx(expected)
    assert factors["reversal_21d"].iloc[-1]["EWY"] == pytest.approx(-expected)


def test_low_risk_factor_orientation_is_reversed() -> None:
    raw = compute_factor_zoo(_frames(), ["realized_vol_20d"])["realized_vol_20d"]
    oriented = orient_factor(raw, "realized_vol_20d")
    pd.testing.assert_frame_equal(oriented, -raw)


def test_future_changes_do_not_alter_past_factor_values() -> None:
    original = _frames()
    option_volume = _option_volume_frames()
    changed = {symbol: frame.copy() for symbol, frame in original.items()}
    for frame in changed.values():
        frame.loc[frame.index[-5:], ["Open", "High", "Low", "Close"]] *= 3
        frame.loc[frame.index[-5:], "Volume"] *= 4

    names = FACTOR_REGISTRY.names()
    before = compute_factor_zoo(original, names, option_volume_frames=option_volume)
    after = compute_factor_zoo(changed, names, option_volume_frames=option_volume)
    cutoff = original["SPY"]["Date"].iloc[-6]
    for name in names:
        pd.testing.assert_frame_equal(before[name].loc[:cutoff], after[name].loc[:cutoff])


def test_factor_coverage_uses_first_non_null_date(tmp_path) -> None:
    frames = _frames(30)
    factors = compute_factor_zoo(frames, ["momentum_21d"])
    write_factor_outputs(
        factors,
        FACTOR_REGISTRY.manifest(["momentum_21d"]),
        tmp_path,
    )
    coverage = pd.read_csv(tmp_path / "factor_coverage.csv")
    expected = frames["SPY"]["Date"].iloc[21].strftime("%Y-%m-%d")
    assert coverage.loc[0, "first_valid_date"] == expected


NEW_FACTOR_NAMES = [
    "high_252d",
    "ts_momentum_252d",
    "vol_managed_momentum_126d",
    "momentum_crash_protected_252d",
    "reversal_63d",
    "downside_beta_60d",
    "co_skew_60d",
    "realized_kurtosis_60d",
    "tail_var_252d",
    "co_kurtosis_60d",
    "beta_bab_60d",
    "roll_spread_20d",
    "corwin_schultz_spread_2d",
    "zero_volume_days_63d",
    "pastor_stambaugh_liquidity_60d",
    "liquidity_commonality_60d",
    "price_delay_60d",
    "intraday_volatility_ratio_20d",
    "macd_histogram_12_26_9",
    "bollinger_position_20d",
    "williams_r_14d",
    "obv_slope_20d",
    "adx_14d",
    "option_volume_trend_21_63d",
    "option_volume_surprise_20d",
    "option_volume_ratio_20d",
    "absorption_ratio_60d",
    "pairwise_correlation_60d",
    "comomentum_60d",
]


def test_new_factors_are_registered() -> None:
    registered = set(FACTOR_REGISTRY.names())
    assert set(NEW_FACTOR_NAMES) <= registered
    assert len(NEW_FACTOR_NAMES) == len(set(NEW_FACTOR_NAMES))


def test_new_factors_compute_cleanly() -> None:
    periods = 360
    frames = _noisy_frames(periods)
    option_volume = {
        symbol: frame[["Date"]].assign(
            OptionVolume=np.abs(np.random.default_rng(hash(symbol) % 2**31).normal(
                50_000, 10_000, periods
            )).clip(min=1.0)
        )
        for symbol, frame in frames.items()
    }
    expected_columns = list(frames)
    factors = compute_factor_zoo(
        frames, NEW_FACTOR_NAMES, option_volume_frames=option_volume
    )

    assert set(factors) == set(NEW_FACTOR_NAMES)
    for name in NEW_FACTOR_NAMES:
        values = factors[name]
        assert values.shape == (periods, len(expected_columns)), name
        assert list(values.columns) == expected_columns, name
        arr = values.to_numpy()
        finite_mask = ~np.isnan(arr)
        assert finite_mask.any(), f"{name} produced only NaN"
        assert np.isfinite(arr[finite_mask]).all(), f"{name} contains inf"
        tail = values.iloc[-60:]
        assert tail.notna().any().any(), f"{name} has no values in the tail"


def test_option_volume_factors_raise_without_option_data() -> None:
    frames = _frames(80)
    option_factors = [
        "option_volume_trend_21_63d",
        "option_volume_surprise_20d",
        "option_volume_ratio_20d",
    ]
    for name in option_factors:
        with pytest.raises(ValueError):
            compute_factor_zoo(frames, [name])


def test_venue_imbalance_helper() -> None:
    periods = 60
    dates = pd.bdate_range("2023-01-02", periods=periods)
    columns = ["EWY", "SPY", "QQQ", "IWM"]
    base = np.arange(periods, dtype=float)
    cboe = pd.DataFrame(
        {col: 1000 + base * 10 + i * 100 for i, col in enumerate(columns)},
        index=dates,
    )
    bats = pd.DataFrame(
        {col: 800 + base * 8 + i * 80 for i, col in enumerate(columns)},
        index=dates,
    )
    edgx = pd.DataFrame(
        {col: 500 + base * 5 + i * 50 for i, col in enumerate(columns)},
        index=dates,
    )
    result = compute_venue_imbalance_20d({"CBOE": cboe, "BATS": bats, "EDGX": edgx})
    assert result.shape == (periods, len(columns))
    assert list(result.columns) == columns
    assert result.iloc[-1].notna().all()
    arr = result.to_numpy()
    finite = arr[~np.isnan(arr)]
    assert (finite >= 0.0).all()
    assert (finite <= 1.0).all()
