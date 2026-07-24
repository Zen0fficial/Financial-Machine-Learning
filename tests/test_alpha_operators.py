from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_factor_screening.alpha_operators import (
    ARITY,
    abs,
    add,
    decay_linear,
    delay,
    delta,
    divide,
    eq,
    greater,
    if_cond,
    less,
    log,
    max_,
    min_,
    multiply,
    power,
    rank,
    scale,
    sign,
    signed_power,
    subtract,
    ts_arg_max,
    ts_arg_min,
    ts_corr,
    ts_cov,
    ts_kurt,
    ts_max,
    ts_mean,
    ts_min,
    ts_quantile,
    ts_rank,
    ts_regression,
    ts_regression_beta,
    ts_regression_intercept,
    ts_regression_residual,
    ts_skew,
    ts_std_dev,
    ts_sum,
    ts_var,
    zscore,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=20)
    symbols = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(rng.normal(10.0, 1.0, size=(20, 5)), index=dates, columns=symbols)


@pytest.fixture
def panel2() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2024-01-02", periods=20)
    symbols = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(rng.normal(5.0, 0.5, size=(20, 5)), index=dates, columns=symbols)


# ---------------------------------------------------------------------------
# Cross-sectional operators
# ---------------------------------------------------------------------------


def test_rank_shape_and_range(panel: pd.DataFrame) -> None:
    result = rank(panel)
    assert result.shape == panel.shape
    valid = result.to_numpy()[~result.isna().to_numpy()]
    assert np.all(valid >= 0)
    assert np.all(valid <= 1)


def test_rank_known_values() -> None:
    df = pd.DataFrame([[1.0, 2.0, 3.0]])
    result = rank(df)
    expected = pd.DataFrame([[1.0 / 3.0, 2.0 / 3.0, 1.0]])
    pd.testing.assert_frame_equal(result, expected)


def test_scale_rows_sum_to_one(panel: pd.DataFrame) -> None:
    result = scale(panel)
    row_sums = result.abs().sum(axis=1)
    valid = row_sums.to_numpy()[~row_sums.isna().to_numpy()]
    assert np.allclose(valid, 1.0)


def test_scale_no_inf(panel: pd.DataFrame) -> None:
    result = scale(panel)
    assert not np.any(np.isinf(result.to_numpy()))


def test_zscore_row_mean_zero(panel: pd.DataFrame) -> None:
    result = zscore(panel)
    valid = result.dropna()
    means = valid.mean(axis=1)
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-9)


def test_zscore_no_inf(panel: pd.DataFrame) -> None:
    result = zscore(panel)
    arr = result.to_numpy()
    assert not np.any(np.isinf(arr[np.isfinite(arr)]))


# ---------------------------------------------------------------------------
# Time-series operators
# ---------------------------------------------------------------------------


def test_delay(panel: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(delay(panel, 2), panel.shift(2))


def test_delta(panel: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(delta(panel, 1), panel - panel.shift(1))


def test_delta_zero_is_zero(panel: pd.DataFrame) -> None:
    result = delta(panel, 0)
    expected = panel - panel
    pd.testing.assert_frame_equal(result, expected)


def test_ts_mean(panel: pd.DataFrame) -> None:
    expected = panel.rolling(5, min_periods=5).mean()
    pd.testing.assert_frame_equal(ts_mean(panel, 5), expected)


def test_ts_sum(panel: pd.DataFrame) -> None:
    expected = panel.rolling(3, min_periods=3).sum()
    pd.testing.assert_frame_equal(ts_sum(panel, 3), expected)


def test_ts_std_dev(panel: pd.DataFrame) -> None:
    expected = panel.rolling(5, min_periods=5).std()
    pd.testing.assert_frame_equal(ts_std_dev(panel, 5), expected)


def test_ts_var(panel: pd.DataFrame) -> None:
    expected = panel.rolling(5, min_periods=5).var()
    pd.testing.assert_frame_equal(ts_var(panel, 5), expected)


def test_ts_min_and_ts_max(panel: pd.DataFrame) -> None:
    n = 4
    pd.testing.assert_frame_equal(
        ts_min(panel, n), panel.rolling(n, min_periods=n).min()
    )
    pd.testing.assert_frame_equal(
        ts_max(panel, n), panel.rolling(n, min_periods=n).max()
    )


def test_ts_rank_range(panel: pd.DataFrame) -> None:
    result = ts_rank(panel, 5)
    assert result.shape == panel.shape
    valid = result.to_numpy()[~result.isna().to_numpy()]
    assert np.all(valid >= 0)
    assert np.all(valid <= 1)


def test_ts_arg_max_known() -> None:
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [5.0, 4.0, 3.0, 2.0, 1.0],
            "C": [1.0, 5.0, 2.0, 3.0, 4.0],
        }
    )
    result = ts_arg_max(df, 5)
    last = result.iloc[-1]
    assert last["A"] == 4.0
    assert last["B"] == 0.0
    assert last["C"] == 1.0


def test_ts_arg_min_known() -> None:
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    result = ts_arg_min(df, 5)
    last = result.iloc[-1]
    assert last["A"] == 0.0
    assert last["B"] == 4.0


def test_ts_corr_matches_pandas(panel: pd.DataFrame, panel2: pd.DataFrame) -> None:
    n = 5
    expected = panel.rolling(n, min_periods=n).corr(panel2)
    pd.testing.assert_frame_equal(ts_corr(panel, panel2, n), expected)


def test_ts_cov_matches_pandas(panel: pd.DataFrame, panel2: pd.DataFrame) -> None:
    n = 5
    expected = panel.rolling(n, min_periods=n).cov(panel2)
    pd.testing.assert_frame_equal(ts_cov(panel, panel2, n), expected)


def test_decay_linear_constant_input() -> None:
    df = pd.DataFrame({"A": [3.0] * 10})
    result = decay_linear(df, 5)
    valid = result.dropna().to_numpy()
    assert np.allclose(valid, 3.0)


def test_decay_linear_weighted_value() -> None:
    # weights = (0, 1, 2, 3, 4) / 10 oldest-to-newest
    # value = 0*1 + 0.1*2 + 0.2*3 + 0.3*4 + 0.4*5 = 4.0
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = decay_linear(df, 5)
    assert result.iloc[-1, 0] == pytest.approx(4.0)


def test_decay_linear_n_equals_one() -> None:
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    pd.testing.assert_frame_equal(decay_linear(df, 1), df)


def test_ts_skew_matches_pandas(panel: pd.DataFrame) -> None:
    expected = panel.rolling(10, min_periods=10).skew()
    pd.testing.assert_frame_equal(ts_skew(panel, 10), expected)


def test_ts_kurt_matches_pandas(panel: pd.DataFrame) -> None:
    expected = panel.rolling(10, min_periods=10).kurt()
    pd.testing.assert_frame_equal(ts_kurt(panel, 10), expected)


def test_ts_quantile_matches_pandas(panel: pd.DataFrame) -> None:
    expected = panel.rolling(5, min_periods=5).quantile(0.5)
    pd.testing.assert_frame_equal(ts_quantile(panel, 5, 0.5), expected)


def test_ts_regression_perfect_linear() -> None:
    n = 30
    x = pd.DataFrame({"A": np.linspace(0.0, 10.0, n)})
    y = 2.0 * x + 1.0  # perfect linear relationship
    intercept, beta, residual = ts_regression(y, x, 10)
    last_intercept = intercept.iloc[-1, 0]
    last_beta = beta.iloc[-1, 0]
    last_resid = residual.iloc[-1, 0]
    assert last_beta == pytest.approx(2.0)
    assert last_intercept == pytest.approx(1.0)
    assert last_resid == pytest.approx(0.0, abs=1e-9)


def test_ts_regression_components_match_tuple() -> None:
    n = 25
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(n, 2)), columns=["A", "B"])
    y = pd.DataFrame(rng.normal(size=(n, 2)), columns=["A", "B"])
    intercept, beta, residual = ts_regression(y, x, 10)
    pd.testing.assert_frame_equal(ts_regression_intercept(y, x, 10), intercept)
    pd.testing.assert_frame_equal(ts_regression_beta(y, x, 10), beta)
    pd.testing.assert_frame_equal(ts_regression_residual(y, x, 10), residual)


def test_ts_regression_no_inf() -> None:
    n = 30
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["A", "B", "C"])
    # Inject a constant column to force zero variance in one regression window.
    x["A"] = 1.0
    y = pd.DataFrame(rng.normal(size=(n, 3)), columns=["A", "B", "C"])
    intercept, beta, residual = ts_regression(y, x, 10)
    for frame in (intercept, beta, residual):
        arr = frame.to_numpy()
        assert not np.any(np.isinf(arr))


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


def test_add_subtract_multiply(panel: pd.DataFrame, panel2: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(add(panel, panel2), panel + panel2)
    pd.testing.assert_frame_equal(subtract(panel, panel2), panel - panel2)
    pd.testing.assert_frame_equal(multiply(panel, panel2), panel * panel2)


def test_divide_safe_for_zero_denominator() -> None:
    a = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"A": [0.0, 1.0, 2.0]})
    result = divide(a, b)
    assert np.isnan(result.iloc[0, 0])
    assert result.iloc[1, 0] == 2.0
    assert result.iloc[2, 0] == 1.5


def test_sign_values() -> None:
    df = pd.DataFrame({"A": [-2.0, 0.0, 3.0]})
    result = sign(df)
    assert result.iloc[0, 0] == -1.0
    assert result.iloc[1, 0] == 0.0
    assert result.iloc[2, 0] == 1.0


def test_abs_values() -> None:
    df = pd.DataFrame({"A": [-2.0, 0.0, 3.0]})
    result = abs(df)
    assert result.iloc[0, 0] == 2.0
    assert result.iloc[1, 0] == 0.0
    assert result.iloc[2, 0] == 3.0


def test_log_handles_non_positive() -> None:
    df = pd.DataFrame({"A": [1.0, 0.0, -1.0, np.e]})
    result = log(df)
    assert result.iloc[0, 0] == pytest.approx(0.0)
    assert np.isnan(result.iloc[1, 0])
    assert np.isnan(result.iloc[2, 0])
    assert result.iloc[3, 0] == pytest.approx(1.0)


def test_power() -> None:
    df = pd.DataFrame({"A": [2.0, 3.0]})
    result = power(df, 2)
    assert result.iloc[0, 0] == 4.0
    assert result.iloc[1, 0] == 9.0


def test_signed_power_preserves_sign() -> None:
    df = pd.DataFrame({"A": [-2.0, 3.0]})
    result = signed_power(df, 2)
    assert result.iloc[0, 0] == -4.0
    assert result.iloc[1, 0] == 9.0


def test_less_greater_eq_with_nan() -> None:
    a = pd.DataFrame({"A": [1.0, 2.0, 3.0, np.nan]})
    b = pd.DataFrame({"A": [2.0, 2.0, 2.0, 2.0]})

    less_result = less(a, b)
    assert less_result.iloc[0, 0] == 1.0
    assert less_result.iloc[1, 0] == 0.0
    assert less_result.iloc[2, 0] == 0.0
    assert np.isnan(less_result.iloc[3, 0])

    greater_result = greater(a, b)
    assert greater_result.iloc[0, 0] == 0.0
    assert greater_result.iloc[1, 0] == 0.0
    assert greater_result.iloc[2, 0] == 1.0
    assert np.isnan(greater_result.iloc[3, 0])

    eq_result = eq(a, b)
    assert eq_result.iloc[0, 0] == 0.0
    assert eq_result.iloc[1, 0] == 1.0
    assert eq_result.iloc[2, 0] == 0.0
    assert np.isnan(eq_result.iloc[3, 0])


def test_max_and_min_elementwise() -> None:
    a = pd.DataFrame({"A": [1.0, 5.0, np.nan]})
    b = pd.DataFrame({"A": [3.0, 2.0, 7.0]})
    max_result = max_(a, b)
    min_result = min_(a, b)
    assert max_result.iloc[0, 0] == 3.0
    assert max_result.iloc[1, 0] == 5.0
    assert np.isnan(max_result.iloc[2, 0])
    assert min_result.iloc[0, 0] == 1.0
    assert min_result.iloc[1, 0] == 2.0
    assert np.isnan(min_result.iloc[2, 0])


def test_if_cond_branches() -> None:
    cond = pd.DataFrame({"A": [1.0, 0.0, -1.0, np.nan]})
    a = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0]})
    b = pd.DataFrame({"A": [100.0, 200.0, 300.0, 400.0]})
    result = if_cond(cond, a, b)
    assert result.iloc[0, 0] == 10.0
    assert result.iloc[1, 0] == 200.0
    assert result.iloc[2, 0] == 300.0
    assert np.isnan(result.iloc[3, 0])


def test_if_cond_with_scalar_branches() -> None:
    cond = pd.DataFrame({"A": [1.0, 0.0, np.nan]})
    a = pd.DataFrame({"A": [5.0, 6.0, 7.0]})
    result = if_cond(cond, a, -1.0)
    assert result.iloc[0, 0] == 5.0
    assert result.iloc[1, 0] == -1.0
    assert np.isnan(result.iloc[2, 0])


def test_if_cond_with_two_scalars() -> None:
    cond = pd.DataFrame({"A": [1.0, 0.0, np.nan]})
    result = if_cond(cond, 10.0, -1.0)
    assert result.iloc[0, 0] == 10.0
    assert result.iloc[1, 0] == -1.0
    assert np.isnan(result.iloc[2, 0])


# ---------------------------------------------------------------------------
# Cross-cutting sanity checks
# ---------------------------------------------------------------------------


def test_all_operators_produce_no_inf(
    panel: pd.DataFrame, panel2: pd.DataFrame
) -> None:
    positive_panel = panel.where(panel > 0)
    results: list[object] = [
        rank(panel),
        scale(panel),
        zscore(panel),
        delay(panel, 3),
        delta(panel, 3),
        ts_mean(panel, 5),
        ts_sum(panel, 5),
        ts_std_dev(panel, 5),
        ts_var(panel, 5),
        ts_min(panel, 5),
        ts_max(panel, 5),
        ts_rank(panel, 5),
        ts_arg_min(panel, 5),
        ts_arg_max(panel, 5),
        ts_corr(panel, panel2, 5),
        ts_cov(panel, panel2, 5),
        decay_linear(panel, 5),
        ts_skew(panel, 10),
        ts_kurt(panel, 10),
        ts_quantile(panel, 5, 0.5),
        sign(panel),
        abs(panel),
        log(positive_panel),
        power(panel, 2),
        signed_power(panel, 2),
        add(panel, panel2),
        subtract(panel, panel2),
        multiply(panel, panel2),
        divide(panel, panel2),
        less(panel, panel2),
        greater(panel, panel2),
        eq(panel, panel2),
        max_(panel, panel2),
        min_(panel, panel2),
        if_cond(less(panel, panel2), panel, panel2),
        ts_regression(panel, panel2, 5),
    ]
    for result in results:
        frames = result if isinstance(result, tuple) else (result,)
        for frame in frames:
            arr = frame.to_numpy()
            finite_mask = np.isfinite(arr)
            assert np.all(finite_mask | np.isnan(arr)), "Found inf in operator output"
            assert not np.any(np.isinf(arr))


def test_arity_registry_has_all_operators() -> None:
    expected_names = {
        "rank",
        "scale",
        "zscore",
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
        "ts_regression_intercept",
        "ts_regression_beta",
        "ts_regression_residual",
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
    }
    assert set(ARITY) == expected_names
    for name, (arity, func) in ARITY.items():
        assert callable(func), f"{name} is not callable"
        assert arity >= 1, f"{name} arity must be positive"
