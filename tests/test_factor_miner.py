"""Tests for the genetic programming factor mining module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gplearn.functions import _Function

from us_factor_screening.factor_miner import (
    MinedFactor,
    MiningConfig,
    _compute_rankic,
    _factor_correlation,
    compute_existing_factors,
    define_function_set,
    evaluate_program,
    format_program,
    load_panel_data,
    make_fitness_function,
    mine_factors,
    prepare_terminals,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "free_nasdaq100_2024_2026"
OHLCV_PATH = DATA_DIR / "ohlcv.csv"
OPTION_VOLUME_PATH = DATA_DIR / "cboe_option_volume_daily.csv.gz"

HAS_REAL_DATA = OHLCV_PATH.exists()


# ---------------------------------------------------------------------------
# Synthetic panel fixture
# ---------------------------------------------------------------------------


def _synthetic_panel(periods: int = 300, n_symbols: int = 20, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Build a synthetic Date x Symbol OHLCV panel for fast unit tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=periods)
    symbols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    price = 100.0 + rng.uniform(0, 50, n_symbols)
    for t in range(periods):
        price = price * (1.0 + rng.normal(0.001, 0.02, n_symbols))
        close.iloc[t] = price
    returns = close.pct_change(fill_method=None)
    panel = {
        "open": close * (1 + rng.normal(0, 0.002, (periods, n_symbols))),
        "high": close * (1 + np.abs(rng.normal(0, 0.01, (periods, n_symbols)))),
        "low": close * (1 - np.abs(rng.normal(0, 0.01, (periods, n_symbols)))),
        "close": close,
        "volume": pd.DataFrame(
            rng.integers(100_000, 1_000_000, (periods, n_symbols)),
            index=dates,
            columns=symbols,
            dtype=float,
        ),
        "returns": returns,
    }
    return panel


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    return _synthetic_panel()


# ---------------------------------------------------------------------------
# load_panel_data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_REAL_DATA, reason="real OHLCV data not available")
def test_load_panel_data_shapes() -> None:
    panel = load_panel_data(str(OHLCV_PATH))
    expected_fields = {"open", "high", "low", "close", "volume", "returns"}
    assert expected_fields.issubset(panel.keys())
    close = panel["close"]
    assert close.shape[0] > 500  # ~2.5 years of trading days
    assert close.shape[1] >= 90  # ~97 Nasdaq-100 symbols
    assert close.index.is_monotonic_increasing
    for field in ("open", "high", "low", "volume"):
        assert panel[field].shape == close.shape
    assert panel["returns"].shape == close.shape


@pytest.mark.skipif(not HAS_REAL_DATA, reason="real OHLCV data not available")
def test_load_panel_data_with_option_volume() -> None:
    panel = load_panel_data(str(OHLCV_PATH), option_volume_path=str(OPTION_VOLUME_PATH))
    assert "option_volume" in panel
    assert panel["option_volume"].shape == panel["close"].shape


@pytest.mark.skipif(not HAS_REAL_DATA, reason="real OHLCV data not available")
def test_load_panel_data_date_filter() -> None:
    panel = load_panel_data(str(OHLCV_PATH), start_date="2025-01-01", end_date="2025-06-30")
    close = panel["close"]
    assert close.index[0] >= pd.Timestamp("2025-01-01")
    assert close.index[-1] <= pd.Timestamp("2025-06-30")


# ---------------------------------------------------------------------------
# compute_existing_factors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_REAL_DATA, reason="real OHLCV data not available")
def test_compute_existing_factors_count() -> None:
    panel = load_panel_data(
        str(OHLCV_PATH), option_volume_path=str(OPTION_VOLUME_PATH)
    )
    factors = compute_existing_factors(panel)
    assert len(factors) == 61
    template = panel["close"]
    for name, values in factors.items():
        assert values.shape == template.shape, f"{name}: shape mismatch"
        assert values.index.equals(template.index), f"{name}: index mismatch"


def test_compute_existing_factors_synthetic() -> None:
    panel = _synthetic_panel(periods=320, n_symbols=12)
    factors = compute_existing_factors(panel)
    # Without option_volume we lose 3 option-volume factors (61 - 3 = 58).
    assert len(factors) >= 55
    template = panel["close"]
    for _name, values in factors.items():
        assert values.shape == template.shape


# ---------------------------------------------------------------------------
# prepare_terminals
# ---------------------------------------------------------------------------


def test_prepare_terminals_includes_ohlcv(panel: dict[str, pd.DataFrame]) -> None:
    terminals = prepare_terminals(panel)
    for field in ("open", "high", "low", "close", "volume", "returns"):
        assert field in terminals
        assert terminals[field].shape == panel["close"].shape


def test_prepare_terminals_with_existing_factors(panel: dict[str, pd.DataFrame]) -> None:
    existing = {"my_factor": panel["close"].pct_change(21)}
    terminals = prepare_terminals(panel, existing)
    assert "my_factor" in terminals
    assert "close" in terminals


# ---------------------------------------------------------------------------
# define_function_set
# ---------------------------------------------------------------------------


def test_define_function_set_basic() -> None:
    functions = define_function_set()
    assert len(functions) >= 30
    for fn in functions:
        assert isinstance(fn, _Function)
        assert isinstance(fn.name, str)
        assert fn.arity in (1, 2)
        assert callable(fn.function)


def test_define_function_set_contains_expected_names() -> None:
    functions = define_function_set()
    names = {fn.name for fn in functions}
    # Cross-sectional / unary
    assert {"rank", "scale", "zscore", "sign", "abs", "log"} <= names
    # Binary arithmetic
    assert {"add", "subtract", "multiply", "divide", "max_", "min_"} <= names
    # Parameterised time-series variants
    assert {"delay_1", "delay_5", "ts_mean_5", "ts_mean_20", "ts_std_dev_60"} <= names
    assert {"ts_corr_5", "ts_corr_10", "ts_corr_20"} <= names
    assert {"decay_linear_5", "ts_rank_20", "ts_arg_max_10"} <= names


def test_function_set_arities_consistent() -> None:
    functions = define_function_set()
    arity_1 = [fn for fn in functions if fn.arity == 1]
    arity_2 = [fn for fn in functions if fn.arity == 2]
    assert len(arity_1) > 0
    assert len(arity_2) > 0
    # Each arity group must have at least 2 members for point mutation to work.
    assert len(arity_1) >= 2
    assert len(arity_2) >= 2


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------


def test_fitness_function_returns_non_negative(panel: dict[str, pd.DataFrame]) -> None:
    h = 1
    forward_returns = panel["close"].shift(-h) / panel["close"] - 1.0
    metric = make_fitness_function(forward_returns)
    factor = panel["close"].pct_change(21)
    y_pred = np.empty(1, dtype=object)
    y_pred[0] = factor
    fitness = metric(np.array([0.0]), y_pred, np.array([1.0]))
    assert isinstance(fitness, float)
    assert fitness >= 0.0


def test_fitness_function_zero_for_non_dataframe(panel: dict[str, pd.DataFrame]) -> None:
    forward_returns = panel["close"].shift(-1) / panel["close"] - 1.0
    metric = make_fitness_function(forward_returns)
    # Numeric y_pred (constant program) -> 0 fitness
    assert metric(np.array([0.0]), np.array([3.0]), np.array([1.0])) == 0.0
    # Object array with a float -> 0 fitness
    y_pred = np.empty(1, dtype=object)
    y_pred[0] = 3.0
    assert metric(np.array([0.0]), y_pred, np.array([1.0])) == 0.0


def test_fitness_function_known_factor(panel: dict[str, pd.DataFrame]) -> None:
    """A 21-day momentum factor should produce a finite, non-zero fitness value."""
    h = 1
    forward_returns = panel["close"].shift(-h) / panel["close"] - 1.0
    metric = make_fitness_function(forward_returns)
    momentum = panel["close"].pct_change(21)
    y_pred = np.empty(1, dtype=object)
    y_pred[0] = momentum
    fitness = metric(np.array([0.0]), y_pred, np.array([1.0]))
    assert np.isfinite(fitness)
    assert fitness > 0.0  # |mean RankIC| should be positive


# ---------------------------------------------------------------------------
# mine_factors
# ---------------------------------------------------------------------------


def test_mine_factors_tiny_config(panel: dict[str, pd.DataFrame]) -> None:
    config = MiningConfig(
        population_size=50,
        generations=3,
        n_factors=5,
        rankic_threshold=0.0,
        correlation_threshold=0.99,
        tournament_size=5,
    )
    mined = mine_factors(panel, config)
    assert len(mined) > 0
    for mf in mined:
        assert isinstance(mf, MinedFactor)
        assert isinstance(mf.formula, str)
        assert len(mf.formula) > 0
        assert np.isfinite(mf.rankic)
        assert np.isfinite(mf.rankic_ir)
        assert np.isfinite(mf.fitness)
        assert mf.program is not None


def test_mine_factors_formula_is_readable(panel: dict[str, pd.DataFrame]) -> None:
    config = MiningConfig(
        population_size=30,
        generations=2,
        n_factors=3,
        rankic_threshold=0.0,
        correlation_threshold=0.99,
        tournament_size=3,
    )
    mined = mine_factors(panel, config)
    assert len(mined) > 0
    for mf in mined:
        # Formula should contain at least one operator or terminal name.
        assert any(c.isalpha() for c in mf.formula)


def test_mine_factors_diversity(panel: dict[str, pd.DataFrame]) -> None:
    """Mined factors should have pairwise |correlation| below the threshold."""
    threshold = 0.7
    config = MiningConfig(
        population_size=80,
        generations=3,
        n_factors=5,
        rankic_threshold=0.0,
        correlation_threshold=threshold,
        tournament_size=5,
    )
    mined = mine_factors(panel, config)
    if len(mined) < 2:
        pytest.skip("Need at least 2 mined factors to test diversity")
    # Reconstruct factor DataFrames from the programs.
    terminals = prepare_terminals(panel)
    terminal_names = list(terminals.keys())
    X = np.empty((1, len(terminal_names)), dtype=object)
    for i, name in enumerate(terminal_names):
        X[0, i] = terminals[name]
    factor_values = []
    for mf in mined:
        factor = evaluate_program(mf.program, X)
        assert factor is not None, f"Failed to re-evaluate: {mf.formula}"
        factor_values.append(factor)
    for i in range(len(factor_values)):
        for j in range(i + 1, len(factor_values)):
            corr = _factor_correlation(factor_values[i], factor_values[j])
            assert abs(corr) <= threshold + 1e-6, (
                f"Factors {i} and {j} have |corr|={abs(corr):.3f} > {threshold}"
            )


def test_mine_factors_reproducible(panel: dict[str, pd.DataFrame]) -> None:
    config = MiningConfig(
        population_size=30,
        generations=2,
        n_factors=3,
        rankic_threshold=0.0,
        correlation_threshold=0.99,
        tournament_size=3,
        random_state=123,
    )
    run1 = mine_factors(panel, config)
    run2 = mine_factors(panel, config)
    assert len(run1) == len(run2)
    for a, b in zip(run1, run2, strict=True):
        assert a.formula == b.formula
        assert abs(a.rankic - b.rankic) < 1e-10


# ---------------------------------------------------------------------------
# format_program
# ---------------------------------------------------------------------------


def test_format_program_returns_string(panel: dict[str, pd.DataFrame]) -> None:
    config = MiningConfig(
        population_size=20,
        generations=1,
        n_factors=1,
        rankic_threshold=0.0,
        correlation_threshold=0.99,
        tournament_size=3,
    )
    mined = mine_factors(panel, config)
    if mined:
        s = format_program(mined[0].program)
        assert isinstance(s, str)
        assert len(s) > 0


# ---------------------------------------------------------------------------
# Integration test with real data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_REAL_DATA, reason="real OHLCV data not available")
def test_mine_factors_real_data_smoke() -> None:
    """End-to-end smoke test on the real Nasdaq-100 panel."""
    panel = load_panel_data(str(OHLCV_PATH))
    config = MiningConfig(
        population_size=60,
        generations=2,
        n_factors=3,
        rankic_threshold=0.0,
        correlation_threshold=0.95,
        tournament_size=5,
    )
    mined = mine_factors(panel, config)
    assert len(mined) > 0
    for mf in mined:
        assert isinstance(mf.formula, str)
        assert np.isfinite(mf.rankic)


@pytest.mark.skipif(not HAS_REAL_DATA, reason="real OHLCV data not available")
def test_compute_rankic_momentum_252d() -> None:
    """momentum_252d should produce a finite RankIC against 1-day forward returns."""
    panel = load_panel_data(str(OHLCV_PATH))
    factors = compute_existing_factors(panel)
    momentum = factors["momentum_252d"]
    h = 1
    forward_returns = panel["close"].shift(-h) / panel["close"] - 1.0
    rankic, ir = _compute_rankic(momentum, forward_returns)
    assert np.isfinite(rankic)
    assert np.isfinite(ir)
