from __future__ import annotations

import numpy as np
import pandas as pd

from us_factor_screening.bivariate_sensitivity import (
    BivariateSensitivityConfig,
    estimate_bivariate_rho,
    resolve_worker_count,
)


def _known_signal(periods: int = 180) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(42)
    signal = np.zeros(periods)
    outcome = np.zeros(periods)
    for time_index in range(1, periods):
        signal[time_index] = 0.55 * signal[time_index - 1] + generator.normal(scale=0.7)
        outcome[time_index] = (
            0.25 * outcome[time_index - 1]
            + 0.85 * signal[time_index - 1]
            + generator.normal(scale=0.4)
        )
    price = 100 * np.exp(np.cumsum(outcome * 0.01))
    return pd.bdate_range("2024-01-02", periods=periods), price, signal


def test_bivariate_rho_recovers_robust_known_signal_and_is_price_scale_invariant() -> None:
    dates, price, signal = _known_signal()
    config = BivariateSensitivityConfig(
        var_lags=2,
        horizons=(1,),
        sensitivity_draws=51,
        min_observations=120,
        workers=1,
        seed=3,
    )

    base = estimate_bivariate_rho(
        pd.DataFrame({"KNOWN": price}, index=dates),
        pd.DataFrame({"KNOWN": signal}, index=dates),
        config,
    )
    scaled = estimate_bivariate_rho(
        pd.DataFrame({"KNOWN": 10 * price}, index=dates),
        pd.DataFrame({"KNOWN": signal}, index=dates),
        config,
    )

    assert base.exclusions.empty
    assert scaled.exclusions.empty
    assert base.rho_star.loc["KNOWN", "rho_h1"] > 0.5
    pd.testing.assert_frame_equal(base.rho_star, scaled.rho_star)


def test_bivariate_rho_rejects_nonpositive_backward_adjusted_prices() -> None:
    dates = pd.bdate_range("2024-01-02", periods=140)
    prices = pd.DataFrame({"BAD": np.linspace(10.0, -1.0, len(dates))}, index=dates)
    signal = pd.DataFrame(
        {"BAD": np.random.default_rng(8).normal(size=len(dates))},
        index=dates,
    )
    config = BivariateSensitivityConfig(
        var_lags=2,
        horizons=(1,),
        sensitivity_draws=11,
        min_observations=100,
        workers=1,
    )

    result = estimate_bivariate_rho(prices, signal, config)

    assert result.rho_star.loc["BAD"].isna().all()
    assert "must be positive" in result.exclusions.loc[0, "error"]


def test_automatic_worker_count_reserves_eight_cpus() -> None:
    assert resolve_worker_count(0, cpu_count=64) == 56
    assert resolve_worker_count(0, cpu_count=4) == 1
    assert resolve_worker_count(12, cpu_count=64) == 12


def test_process_parallel_results_match_sequential_results() -> None:
    dates, price, signal = _known_signal(periods=150)
    prices = pd.DataFrame({"ONE": price, "TWO": 1.5 * price}, index=dates)
    signals = pd.DataFrame({"ONE": signal, "TWO": signal}, index=dates)
    common = {
        "var_lags": 2,
        "horizons": (1,),
        "sensitivity_draws": 11,
        "min_observations": 120,
        "seed": 9,
    }

    sequential = estimate_bivariate_rho(
        prices,
        signals,
        BivariateSensitivityConfig(**common, workers=1),
    )
    parallel = estimate_bivariate_rho(
        prices,
        signals,
        BivariateSensitivityConfig(**common, workers=2),
    )

    pd.testing.assert_frame_equal(sequential.rho_star, parallel.rho_star)
    pd.testing.assert_frame_equal(sequential.exclusions, parallel.exclusions)
