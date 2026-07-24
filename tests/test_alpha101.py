from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_factor_screening.alpha101 import ALPHA_101, compute_alpha_101

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv() -> dict[str, pd.DataFrame]:
    """Synthetic OHLCV panels large enough to warm up every alpha.

    600 trading days covers the 250-day look-back inside Alpha#19. Five
    synthetic tickers give enough cross-sectional variation for rank/scale
    operators to be meaningful.
    """
    rng = np.random.default_rng(2024)
    periods = 600
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    dates = pd.bdate_range("2024-01-02", periods=periods)

    daily_returns = pd.DataFrame(
        rng.normal(0.0004, 0.02, size=(periods, len(symbols))),
        index=dates,
        columns=symbols,
    )
    close = 100.0 * (1.0 + daily_returns).cumprod()
    open_ = close.shift(1).fillna(100.0) * (
        1.0 + rng.normal(0.0, 0.005, size=(periods, len(symbols)))
    )
    high = np.maximum(open_, close) * (
        1.0 + np.abs(rng.normal(0.0, 0.005, size=(periods, len(symbols))))
    )
    low = np.minimum(open_, close) * (
        1.0 - np.abs(rng.normal(0.0, 0.005, size=(periods, len(symbols))))
    )
    volume = pd.DataFrame(
        1_000_000.0 + np.abs(rng.normal(0.0, 500_000.0, size=(periods, len(symbols)))),
        index=dates,
        columns=symbols,
    )
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "returns": daily_returns,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_registry_contains_expected_alphas() -> None:
    expected = {
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
    }
    assert set(ALPHA_101) == expected
    assert len(ALPHA_101) >= 20


@pytest.mark.parametrize("alpha_name", list(ALPHA_101))
def test_alpha_returns_correct_shape(
    ohlcv: dict[str, pd.DataFrame], alpha_name: str
) -> None:
    func = ALPHA_101[alpha_name]
    result = func(
        ohlcv["open"],
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        ohlcv["volume"],
        ohlcv["returns"],
    )
    expected_shape = ohlcv["close"].shape
    assert isinstance(result, pd.DataFrame)
    assert result.shape == expected_shape
    pd.testing.assert_index_equal(result.index, ohlcv["close"].index)
    pd.testing.assert_index_equal(result.columns, ohlcv["close"].columns)


@pytest.mark.parametrize("alpha_name", list(ALPHA_101))
def test_alpha_has_no_inf(
    ohlcv: dict[str, pd.DataFrame], alpha_name: str
) -> None:
    func = ALPHA_101[alpha_name]
    result = func(
        ohlcv["open"],
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        ohlcv["volume"],
        ohlcv["returns"],
    )
    arr = result.to_numpy()
    assert not np.any(np.isinf(arr)), f"{alpha_name} produced inf values"


@pytest.mark.parametrize("alpha_name", list(ALPHA_101))
def test_alpha_produces_finite_values_mid_sample(
    ohlcv: dict[str, pd.DataFrame], alpha_name: str
) -> None:
    func = ALPHA_101[alpha_name]
    result = func(
        ohlcv["open"],
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        ohlcv["volume"],
        ohlcv["returns"],
    )
    n_periods = result.shape[0]
    # Mid sample is well past every look-back (max look-back is 250 days).
    mid = n_periods // 2
    window = result.iloc[mid - 5 : mid + 5]
    assert window.notna().any().any(), (
        f"{alpha_name} produced no finite values around mid-sample"
    )


def test_compute_alpha_101_returns_all_panels(ohlcv: dict[str, pd.DataFrame]) -> None:
    panels = compute_alpha_101(ohlcv)
    assert set(panels) == set(ALPHA_101)
    for name, frame in panels.items():
        assert frame.shape == ohlcv["close"].shape
        arr = frame.to_numpy()
        assert not np.any(np.isinf(arr)), f"{name} has inf"


def test_compute_alpha_101_supports_subset(ohlcv: dict[str, pd.DataFrame]) -> None:
    selected = ["alpha_001", "alpha_044"]
    panels = compute_alpha_101(ohlcv, selected)
    assert set(panels) == set(selected)


def test_compute_alpha_101_rejects_unknown_name(
    ohlcv: dict[str, pd.DataFrame],
) -> None:
    with pytest.raises(KeyError):
        compute_alpha_101(ohlcv, ["alpha_999"])


def test_compute_alpha_101_rejects_missing_keys() -> None:
    with pytest.raises(KeyError):
        compute_alpha_101({"open": pd.DataFrame()})


def test_compute_alpha_101_default_returns_match_explicit(
    ohlcv: dict[str, pd.DataFrame],
) -> None:
    """When ``returns`` is not supplied it should be derived from close."""
    frames = {k: v for k, v in ohlcv.items() if k != "returns"}
    explicit = compute_alpha_101(ohlcv, ["alpha_001"])
    derived = compute_alpha_101(frames, ["alpha_001"])
    pd.testing.assert_frame_equal(explicit["alpha_001"], derived["alpha_001"])
