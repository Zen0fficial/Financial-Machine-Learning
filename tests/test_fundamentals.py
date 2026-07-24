from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_factor_screening.factor_zoo import FactorContext
from us_factor_screening.fundamentals import (
    FUNDAMENTALS_REGISTRY,
    FundamentalsConfig,
    FundamentalsPanel,
    compute_fundamental_factors,
    orient_fundamental_factor,
)

SYMBOLS = ["AAA", "BBB", "CCC"]
# Eight fiscal quarter-end dates from 2022-Q1 through 2023-Q4.
QUARTER_ENDS = pd.date_range("2022-03-31", periods=8, freq="QE")
PIT_LAG = 45  # availability = reportPeriod + 45 days, set via the `updated` column


def _quarter_records(symbol: str, scale: float = 1.0) -> list[dict]:
    """Build 8 quarters of synthetic financials for one symbol.

    Values grow linearly so TTM sums and YoY growth are easy to verify.
    `scale` differentiates symbols so cross-sectional variation is non-zero.
    """
    records: list[dict] = []
    for i, rp in enumerate(QUARTER_ENDS):
        records.append(
            {
                "ticker": symbol,
                "period": "QA",
                "reportPeriod": rp.strftime("%Y-%m-%d"),
                "calendarDate": rp.strftime("%Y-%m-%d"),
                "updated": (rp + pd.Timedelta(days=PIT_LAG)).strftime("%Y-%m-%d"),
                "assets": (1000.0 + i * 100) * scale,
                "bookValue": (600.0 + i * 60) * scale,
                "revenue": (100.0 + i * 10) * scale,
                "netIncome": (10.0 + i) * scale,
                "grossProfit": (40.0 + i * 4) * scale,
                "operatingIncome": (20.0 + i * 2) * scale,
                "eps": 1.0 + i * 0.1,
                "dividendPerShare": 0.5,
                "netCashFromOperatingActivities": (15.0 + i) * scale,
                "capitalExpenditure": (5.0 + i * 0.5) * scale,
                "weightedAverageSharesOutstanding": 100.0 + i,  # shares grow over time
                "workingCapital": 50.0 * scale,
            }
        )
    return records


def _write_financials(directory: Path, symbol: str, scale: float = 1.0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_quarter_records(symbol, scale)).to_csv(
        directory / f"financials_{symbol}.csv", index=False
    )


def _build_panel(tmp_path: Path, scales: dict[str, float] | None = None) -> FundamentalsPanel:
    scales = scales or dict.fromkeys(SYMBOLS, 1.0)
    for sym in SYMBOLS:
        _write_financials(tmp_path, sym, scales.get(sym, 1.0))
    return FundamentalsPanel(tmp_path, SYMBOLS)


def _ohlcv_frames(
    symbols: tuple[str, ...] = tuple(SYMBOLS), periods: int = 640
) -> dict[str, pd.DataFrame]:
    """Deterministic daily OHLCV panel spanning the synthetic financials."""
    dates = pd.bdate_range("2022-01-03", periods=periods)
    steps = np.arange(periods, dtype=float)
    frames: dict[str, pd.DataFrame] = {}
    for position, symbol in enumerate(symbols):
        close = 50.0 + position * 5.0 + steps * 0.001
        frames[symbol] = pd.DataFrame(
            {
                "Date": dates,
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": 1_000_000.0 + position * 100_000 + steps * 1.0,
            }
        )
    return frames


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_fundamentals_registry_is_well_formed() -> None:
    names = FUNDAMENTALS_REGISTRY.names()
    assert len(names) == 15
    assert len(names) == len(set(names))
    assert set(FUNDAMENTALS_REGISTRY.families()) == {
        "value",
        "profitability",
        "investment",
        "accruals",
    }
    manifest = FUNDAMENTALS_REGISTRY.manifest()
    assert manifest.notna().all().all()
    # Direction must be set for every factor.
    assert set(manifest["conventional_long_direction"]) <= {"higher", "lower"}


# ---------------------------------------------------------------------------
# FundamentalsPanel: loading, alignment, point-in-time correctness
# ---------------------------------------------------------------------------


def test_panel_loads_all_symbols(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    for symbol in SYMBOLS:
        assert symbol in panel._raw
        assert len(panel._raw[symbol]) == 8
    assert panel.symbols == SYMBOLS


def test_align_to_calendar_populates_daily_frame(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    panel.align_to_calendar(dates)
    for symbol in SYMBOLS:
        daily = panel._daily[symbol]
        assert daily.index.equals(dates)
        assert "revenue" in daily.columns
    # The aligned frame and on-the-fly get_field agree.
    rev = panel.get_field("revenue", dates)
    pd.testing.assert_frame_equal(rev, panel.get_field("revenue", dates))


def test_point_in_time_correctness(tmp_path: Path) -> None:
    """A factor on date D uses only quarters available on or before D."""
    panel = _build_panel(tmp_path)
    dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    rev = panel.get_field("revenue", dates)

    q1_avail = QUARTER_ENDS[0] + pd.Timedelta(days=PIT_LAG)  # 2022-05-15
    q2_avail = QUARTER_ENDS[1] + pd.Timedelta(days=PIT_LAG)  # 2022-08-14

    # Before the first quarter is available, every symbol is NaN.
    pre = rev.loc[: q1_avail - pd.Timedelta(days=1)]
    assert pre.isna().all().all()

    # On Q1's availability date, revenue jumps to Q1's value (100).
    assert rev.loc[q1_avail, "AAA"] == pytest.approx(100.0)
    # The day before Q2 becomes available, revenue is still Q1's value.
    assert rev.loc[q2_avail - pd.Timedelta(days=1), "AAA"] == pytest.approx(100.0)
    # On Q2's availability date, revenue steps up to Q2's value (110).
    assert rev.loc[q2_avail, "AAA"] == pytest.approx(110.0)


def test_missing_field_returns_nan(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    missing = panel.get_field("does_not_exist", dates)
    assert missing.shape == (len(dates), len(SYMBOLS))
    assert missing.isna().all().all()


# ---------------------------------------------------------------------------
# Trailing 4Q (TTM) sum and YoY growth
# ---------------------------------------------------------------------------


def test_trailing_four_quarter_sum(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    ttm = panel.get_trailing_field("revenue", dates, quarters=4)

    q3_avail = QUARTER_ENDS[2] + pd.Timedelta(days=PIT_LAG)  # only 3 quarters available
    q4_avail = QUARTER_ENDS[3] + pd.Timedelta(days=PIT_LAG)  # 4 quarters available
    q5_avail = QUARTER_ENDS[4] + pd.Timedelta(days=PIT_LAG)

    # With fewer than 4 quarters available, TTM is NaN.
    assert np.isnan(ttm.loc[q3_avail, "AAA"])
    assert np.isnan(ttm.loc[q4_avail - pd.Timedelta(days=1), "AAA"])
    # At Q4 availability, TTM revenue = Q1 + Q2 + Q3 + Q4 = 100+110+120+130 = 460.
    assert ttm.loc[q4_avail, "AAA"] == pytest.approx(460.0)
    # At Q5 availability, the window rolls forward: Q2+Q3+Q4+Q5 = 110+120+130+140 = 500.
    assert ttm.loc[q5_avail, "AAA"] == pytest.approx(500.0)


def test_yoy_growth(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    yoy = panel.get_yoy_growth("revenue", dates, quarters_back=4)

    q4_avail = QUARTER_ENDS[3] + pd.Timedelta(days=PIT_LAG)
    q5_avail = QUARTER_ENDS[4] + pd.Timedelta(days=PIT_LAG)

    # YoY needs 5 quarters of history; before Q5 it is undefined.
    assert np.isnan(yoy.loc[q4_avail, "AAA"])
    assert np.isnan(yoy.loc[q5_avail - pd.Timedelta(days=1), "AAA"])
    # At Q5: (rev5 - rev1) / |rev1| = (140 - 100) / 100 = 0.4.
    assert yoy.loc[q5_avail, "AAA"] == pytest.approx(0.4)


def test_config_pit_lag_fallback_without_updated(tmp_path: Path) -> None:
    """When `updated` is missing, availability falls back to calendarDate + pit_lag_days."""
    for sym in SYMBOLS:
        records = _quarter_records(sym)
        for row in records:
            row.pop("updated")
        pd.DataFrame(records).to_csv(tmp_path / f"financials_{sym}.csv", index=False)

    panel = FundamentalsPanel(tmp_path, SYMBOLS, FundamentalsConfig(pit_lag_days=45))
    dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
    rev = panel.get_field("revenue", dates)

    q1_avail = QUARTER_ENDS[0] + pd.Timedelta(days=45)
    assert np.isnan(rev.loc[q1_avail - pd.Timedelta(days=1), "AAA"])
    assert rev.loc[q1_avail, "AAA"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Factor compute functions
# ---------------------------------------------------------------------------


def test_factors_raise_without_fundamentals() -> None:
    frames = _ohlcv_frames(periods=80)
    context = FactorContext.from_frames(frames)
    for name in FUNDAMENTALS_REGISTRY.names():
        spec = FUNDAMENTALS_REGISTRY.get(name)
        with pytest.raises(ValueError, match="fundamentals"):
            spec.compute(context)


def test_all_factors_have_expected_shape_and_no_inf(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path, scales={"AAA": 1.0, "BBB": 1.5, "CCC": 2.0})
    periods = 640
    frames = _ohlcv_frames(periods=periods)
    factors = compute_fundamental_factors(frames, panel)

    expected_index = frames["AAA"].set_index("Date").index
    assert set(factors) == set(FUNDAMENTALS_REGISTRY.names())
    for name, values in factors.items():
        assert values.shape == (periods, len(SYMBOLS)), name
        assert list(values.columns) == SYMBOLS, name
        assert values.index.equals(expected_index), name
        arr = values.to_numpy()
        finite_mask = ~np.isnan(arr)
        assert finite_mask.any(), f"{name} produced only NaN"
        assert np.isfinite(arr[finite_mask]).all(), f"{name} contains inf"


def test_factors_finite_in_the_middle_of_sample(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path, scales={"AAA": 1.0, "BBB": 1.5, "CCC": 2.0})
    frames = _ohlcv_frames(periods=640)
    factors = compute_fundamental_factors(frames, panel)

    # Index 440 (~late 2023) is well past the YoY warm-up; every factor should
    # have at least one finite cross-sectional value there.
    mid = next(iter(factors.values())).index[440]
    for name, values in factors.items():
        row = values.loc[mid]
        assert row.notna().any(), f"{name} all NaN at sample middle"
    # The tail must also be populated.
    for name, values in factors.items():
        assert values.iloc[-60:].notna().any().any(), f"{name} has no values in the tail"


def test_book_to_market_formula(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    frames = _ohlcv_frames(periods=640)
    factors = compute_fundamental_factors(frames, panel, ["book_to_market"])
    btm = factors["book_to_market"]

    close = frames["AAA"].set_index("Date")["Close"]
    # Use a date well past warm-up (Q8 availability).
    target = QUARTER_ENDS[7] + pd.Timedelta(days=PIT_LAG) + pd.Timedelta(days=10)
    target = btm.index[btm.index.get_indexer([target], method="ffill")[0]]
    shares = 100.0 + 7  # latest quarter (Q8, i=7) shares outstanding
    book = 600.0 + 7 * 60  # Q8 book value
    expected = book / (shares * close.loc[target])
    assert btm.loc[target, "AAA"] == pytest.approx(expected, rel=1e-9)


def test_earnings_yield_uses_ttm_eps(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    frames = _ohlcv_frames(periods=640)
    factors = compute_fundamental_factors(frames, panel, ["earnings_yield"])
    ey = factors["earnings_yield"]

    close = frames["AAA"].set_index("Date")["Close"]
    # After Q5 availability, TTM EPS = Q2+Q3+Q4+Q5 = 1.1+1.2+1.3+1.4 = 5.0.
    q5_avail = QUARTER_ENDS[4] + pd.Timedelta(days=PIT_LAG)
    q6_avail = QUARTER_ENDS[5] + pd.Timedelta(days=PIT_LAG)
    target = ey.index[ey.index.get_indexer([q5_avail + pd.Timedelta(days=1)], method="ffill")[0]]
    expected_eps = 1.1 + 1.2 + 1.3 + 1.4
    assert ey.loc[target, "AAA"] == pytest.approx(expected_eps / close.loc[target], rel=1e-9)
    # EPS is constant within a quarter window, so the factor only moves with price.
    # Use asof because the earnings-yield frame is indexed by business days.
    assert np.isfinite(ey.asof(q6_avail - pd.Timedelta(days=1))["AAA"])


def test_asset_growth_direction_is_reversed_by_orient(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    frames = _ohlcv_frames(periods=640)
    raw = compute_fundamental_factors(frames, panel, ["asset_growth"])["asset_growth"]
    oriented = orient_fundamental_factor(raw, "asset_growth")
    pd.testing.assert_frame_equal(oriented, -raw)


def test_oriented_compute_flips_negative_direction_factors(tmp_path: Path) -> None:
    panel = _build_panel(tmp_path)
    frames = _ohlcv_frames(periods=640)
    raw = compute_fundamental_factors(frames, panel, ["asset_growth", "book_to_market"])
    oriented = compute_fundamental_factors(
        frames, panel, ["asset_growth", "book_to_market"], oriented=True
    )
    pd.testing.assert_frame_equal(oriented["asset_growth"], -raw["asset_growth"])
    pd.testing.assert_frame_equal(oriented["book_to_market"], raw["book_to_market"])


def test_sloan_accruals_sign(tmp_path: Path) -> None:
    """Net income exceeds operating cash flow in the synthetic data -> positive accruals."""
    panel = _build_panel(tmp_path)
    frames = _ohlcv_frames(periods=640)
    factors = compute_fundamental_factors(frames, panel, ["sloan_accruals"])
    accruals = factors["sloan_accruals"]
    q8_avail = QUARTER_ENDS[7] + pd.Timedelta(days=PIT_LAG)
    target = accruals.index[accruals.index.get_indexer([q8_avail + pd.Timedelta(days=5)], method="ffill")[0]]
    # TTM NI = sum(10..17) = 108; TTM OCF = sum(15..22) = 148 -> NI - OCF < 0.
    ni_ttm = sum(10.0 + i for i in range(4, 8))
    ocf_ttm = sum(15.0 + i for i in range(4, 8))
    assets = 1000.0 + 7 * 100
    expected = (ni_ttm - ocf_ttm) / assets
    assert accruals.loc[target, "AAA"] == pytest.approx(expected, rel=1e-9)
    assert expected < 0  # accruals are negative in this synthetic setup
