"""Tests for the local AlphaGen adapter.

These tests are hermetic (they build synthetic OHLCV panels) so they do not
depend on the bundled Nasdaq-100 CSV and run in a few seconds on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

# Importing the adapter adds AlphaGen to sys.path and bypasses its Qlib hook,
# so it must be imported before the direct alphagen imports below.
# ruff: noqa: I001
from us_factor_screening.alphagen_adapter import (  # noqa: E402
    ALPHAGEN_OPERATORS,
    OPERATOR_TO_LOCAL,
    LocalAlphaCalculator,
    build_environment,
    build_stock_data,
    run_alphagen,
)

from alphagen.data.expression import (  # noqa: E402
    Add,
    CSRank,
    Feature,
    Mean,
    Rank,
)
from alphagen_qlib.stock_data import FeatureType  # noqa: E402

# StockData needs >= max_backtrack_days + max_future_days rows to expose a
# non-empty effective window. Keep the panel comfortably above that threshold.
_N_DAYS = 200
_N_STOCKS = 5
_BACKTRACK = 100
_HORIZON = 20


def _synthetic_panel(
    n_days: int = _N_DAYS, n_stocks: int = _N_STOCKS, seed: int = 0
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    symbols = [f"S{i}" for i in range(n_stocks)]
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(n_days, n_stocks)), axis=0)),
        index=dates,
        columns=symbols,
    )
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.01, size=close.shape)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.01, size=close.shape)))
    opn = close.shift(1).fillna(close)
    volume = pd.DataFrame(rng.integers(1_000, 1_000_000, size=close.shape), index=dates, columns=symbols).astype(float)
    data = {"open": opn, "high": high, "low": low, "close": close, "volume": volume}
    forward_returns = (close.shift(-_HORIZON) / close - 1).iloc[:-_HORIZON]
    return data, forward_returns


@pytest.fixture(scope="module")
def panel() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    return _synthetic_panel()


# ---------------------------------------------------------------------------
# Operator vocabulary
# ---------------------------------------------------------------------------


def test_operator_vocabulary_non_empty() -> None:
    assert len(ALPHAGEN_OPERATORS) >= 20
    # Each entry must be a concrete AlphaGen Operator subclass.
    from alphagen.data.expression import Operator

    for op in ALPHAGEN_OPERATORS:
        assert issubclass(op, Operator)


def test_operator_to_local_map_covers_vocabulary() -> None:
    # Every AlphaGen operator we expose should appear in the translation map
    # (value may be None when no local equivalent exists).
    for op in ALPHAGEN_OPERATORS:
        assert op.__name__ in OPERATOR_TO_LOCAL


# ---------------------------------------------------------------------------
# StockData builder
# ---------------------------------------------------------------------------


def test_build_stock_data_shape_and_padding(panel) -> None:
    data, _ = panel
    sd = build_stock_data(
        data, max_backtrack_days=_BACKTRACK, forward_horizon=_HORIZON
    )
    expected_total = _N_DAYS
    assert sd.data.shape == (expected_total, 6, _N_STOCKS)
    assert sd.n_days == _N_DAYS - _BACKTRACK - _HORIZON
    assert sd.n_stocks == _N_STOCKS
    assert sd.max_backtrack_days == _BACKTRACK
    assert sd.max_future_days == _HORIZON


def test_stock_data_is_qlib_free(panel) -> None:
    """The Qlib bypass flag must remain set after construction."""
    import alphagen_qlib.stock_data as sd_module

    assert sd_module._QLIB_INITIALIZED is True
    build_stock_data(panel[0], max_backtrack_days=_BACKTRACK, forward_horizon=_HORIZON)
    assert sd_module._QLIB_INITIALIZED is True


# ---------------------------------------------------------------------------
# LocalAlphaCalculator
# ---------------------------------------------------------------------------


def test_calc_single_IC_ret_returns_float(panel) -> None:
    data, fwd = panel
    calc = LocalAlphaCalculator(
        data, fwd, max_backtrack_days=_BACKTRACK, forward_horizon=_HORIZON
    )
    close = Feature(FeatureType.CLOSE)
    ic = calc.calc_single_IC_ret(Rank(close, 10))
    assert isinstance(ic, float)
    assert np.isfinite(ic)


def test_calc_single_all_ret_returns_pair(panel) -> None:
    data, fwd = panel
    calc = LocalAlphaCalculator(
        data, fwd, max_backtrack_days=_BACKTRACK, forward_horizon=_HORIZON
    )
    close = Feature(FeatureType.CLOSE)
    ic, ric = calc.calc_single_all_ret(Mean(close, 5))
    assert isinstance(ic, float) and isinstance(ric, float)
    assert np.isfinite(ic) and np.isfinite(ric)


def test_calc_mutual_IC_in_range(panel) -> None:
    data, fwd = panel
    calc = LocalAlphaCalculator(
        data, fwd, max_backtrack_days=_BACKTRACK, forward_horizon=_HORIZON
    )
    close = Feature(FeatureType.CLOSE)
    mic = calc.calc_mutual_IC(Rank(close, 10), Mean(close, 10))
    assert isinstance(mic, float)
    assert -1.01 <= mic <= 1.01


def test_calc_pool_IC_ret_returns_float(panel) -> None:
    data, fwd = panel
    calc = LocalAlphaCalculator(
        data, fwd, max_backtrack_days=_BACKTRACK, forward_horizon=_HORIZON
    )
    close = Feature(FeatureType.CLOSE)
    ic = calc.calc_pool_IC_ret([Rank(close, 10), CSRank(close)], [0.5, 0.5])
    assert isinstance(ic, float)
    assert np.isfinite(ic)


def test_target_shape_matches_n_days(panel) -> None:
    data, fwd = panel
    calc = LocalAlphaCalculator(
        data, fwd, max_backtrack_days=_BACKTRACK, forward_horizon=_HORIZON
    )
    assert calc.target.shape[0] == calc.n_days
    assert calc.target.shape[1] == _N_STOCKS


def test_evaluate_alpha_shape(panel) -> None:
    data, fwd = panel
    calc = LocalAlphaCalculator(
        data, fwd, max_backtrack_days=_BACKTRACK, forward_horizon=_HORIZON
    )
    close = Feature(FeatureType.CLOSE)
    value = calc.evaluate_alpha(Add(close, close))
    assert value.shape == (calc.n_days, _N_STOCKS)


# ---------------------------------------------------------------------------
# RL environment
# ---------------------------------------------------------------------------


def test_environment_creation_and_step(panel) -> None:
    data, fwd = panel
    env, pool, calc = build_environment(
        data=data,
        forward_returns=fwd,
        pool_capacity=5,
    )
    assert env.action_space.n > 0
    assert env.observation_space.shape == (15,)

    obs, info = env.reset(seed=0)
    assert obs.shape == (15,)
    masks = env.action_masks()
    assert masks.shape == (env.action_space.n,)
    assert masks.any()

    rng = np.random.default_rng(0)
    done = False
    step = 0
    while not done and step < 30:
        valid = np.where(env.action_masks())[0]
        action = int(rng.choice(valid))
        obs, reward, done, truncated, info = env.step(action)
        assert np.isfinite(reward)
        step += 1
    assert done


# ---------------------------------------------------------------------------
# Full training smoke test (kept very small)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0])
def test_run_alphagen_smoke(seed: int) -> None:
    from us_factor_screening.alphagen_adapter import _DEFAULT_FORWARD_HORIZON  # noqa: F401

    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "free_nasdaq100_2024_2026"
        / "ohlcv.csv"
    )
    if not data_path.exists():
        pytest.skip("free Nasdaq-100 OHLCV bundle not available")
    result = run_alphagen(
        steps=500,
        max_stocks=8,
        seed=seed,
        verbose=0,
    )
    assert "n_factors" in result
    assert "expressions" in result
    assert "best_ensemble_ic" in result
    assert isinstance(result["expressions"], list)
    assert isinstance(result["best_ensemble_ic"], float)
