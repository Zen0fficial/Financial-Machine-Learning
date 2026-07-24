from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from us_factor_screening.causal_screening import (
    ScreeningConfig,
    _multivariate_granger_f_statistic,
    _panel_granger_f_statistic,
    _stationary_factor_panel,
    forward_returns,
    make_walk_forward_folds,
    screen_factor_horizons,
    screen_factors,
    walk_forward_causal_screen,
)


def test_panel_granger_matches_analysis_notebook_for_one_series() -> None:
    generator = np.random.default_rng(5)
    outcome = generator.normal(size=140)
    signal = generator.normal(size=140)
    lags = 4
    horizon = 1
    notebook_data = pd.DataFrame(
        {"outcome": pd.Series(outcome).shift(-(horizon - 1)), "signal": signal}
    ).dropna()
    expected = grangercausalitytests(notebook_data[["outcome", "signal"]], maxlag=[lags])[lags][0][
        "ssr_ftest"
    ]

    statistic, p_value, _ = _panel_granger_f_statistic(
        outcome[:, None], signal[:, None], lags=lags, horizon=horizon
    )

    assert np.isclose(statistic, expected[0])
    assert np.isclose(p_value, expected[1])

    multivariate_statistic, multivariate_p, _ = _multivariate_granger_f_statistic(
        outcome[:, None],
        signal[:, None],
        lags=lags,
        horizon=horizon,
        common_factors=3,
    )
    assert np.isclose(multivariate_statistic, expected[0])
    assert np.isclose(multivariate_p, expected[1])


def test_multivariate_granger_detects_cross_asset_signal_path() -> None:
    generator = np.random.default_rng(123)
    periods = 260
    assets = 12
    latent_signal = np.zeros(periods)
    for time_index in range(1, periods):
        latent_signal[time_index] = (
            0.7 * latent_signal[time_index - 1] + generator.normal()
        )

    signal = generator.normal(scale=0.6, size=(periods, assets))
    signal[:, :6] += latent_signal[:, None] * np.linspace(0.7, 1.2, 6)
    outcome = generator.normal(scale=0.5, size=(periods, assets))
    target_loadings = np.linspace(0.7, 1.2, 6)
    for time_index in range(1, periods):
        outcome[time_index, :6] = (
            0.2 * outcome[time_index - 1, :6]
            + generator.normal(scale=0.5, size=6)
        )
        outcome[time_index, 6:] = (
            0.2 * outcome[time_index - 1, 6:]
            + 0.8 * latent_signal[time_index - 1] * target_loadings
            + generator.normal(scale=0.5, size=6)
        )

    multivariate_f, multivariate_p, _ = _multivariate_granger_f_statistic(
        outcome,
        signal,
        lags=2,
        horizon=1,
        common_factors=2,
    )
    pooled_f, pooled_p, _ = _panel_granger_f_statistic(
        outcome,
        signal,
        lags=2,
        horizon=1,
    )

    assert multivariate_f > 100
    assert multivariate_p < 1e-12
    assert pooled_f < 1
    assert pooled_p > 0.10


def test_stationarity_test_differences_prices_but_keeps_stationary_signal_levels() -> None:
    generator = np.random.default_rng(19)
    dates = pd.bdate_range("2022-01-03", periods=220)
    symbols = [f"S{index:02d}" for index in range(10)]
    returns = generator.normal(scale=0.01, size=(len(dates), len(symbols)))
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=symbols,
    )
    signal_values = np.zeros_like(returns)
    for time_index in range(1, len(dates)):
        signal_values[time_index] = (
            0.55 * signal_values[time_index - 1]
            + generator.normal(size=len(symbols))
        )
    signal = pd.DataFrame(signal_values, index=dates, columns=symbols)
    config = ScreeningConfig(
        forward_horizon=1,
        effect_horizons=(1,),
        var_lags=2,
        permutations=9,
        sensitivity_draws=0,
    )

    _, _, outcome_panel, signal_panel = _stationary_factor_panel(
        signal,
        close,
        dates,
        config,
    )

    assert outcome_panel.difference_order == 1
    assert signal_panel.difference_order == 0
    assert outcome_panel.final_stationary_fraction >= 0.80
    assert signal_panel.final_stationary_fraction >= 0.80


def test_horizon_profile_recovers_three_session_effect_lifetime() -> None:
    generator = np.random.default_rng(29)
    dates = pd.bdate_range("2022-01-03", periods=240)
    symbols = [f"S{index:02d}" for index in range(12)]
    raw_signal = pd.DataFrame(
        generator.normal(size=(len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )
    ranked_signal = raw_signal.rank(axis=1, pct=True).to_numpy() - 0.5
    returns = generator.normal(scale=0.008, size=ranked_signal.shape)
    for time_index in range(3, len(dates)):
        returns[time_index] += (
            0.08 * ranked_signal[time_index - 1]
            + 0.06 * ranked_signal[time_index - 2]
            + 0.04 * ranked_signal[time_index - 3]
        )
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=symbols,
    )
    config = ScreeningConfig(
        forward_horizon=1,
        effect_horizons=(1, 2, 3, 4),
        var_lags=3,
        permutations=99,
        simulation_batch_size=8,
        q_threshold=0.20,
        granger_p_threshold=0.05,
        min_abs_rank_ic=0.0,
        min_sign_consistency=0.0,
        sensitivity_draws=0,
        min_sensitivity_rho=0.0,
        min_dates=100,
    )

    result = screen_factor_horizons(
        {"three_day_signal": raw_signal},
        close,
        dates,
        config,
    )
    metrics = result.metrics.set_index("factor")
    horizons = result.horizon_metrics.set_index("horizon")

    assert metrics.loc["three_day_signal", "robust_effect_horizon_sessions"] == 3, horizons[
        ["granger_p_value", "fisher_p_value", "fisher_q_value", "robust_effective"]
    ]
    assert horizons.loc[1:3, "robust_effective"].all()
    assert not bool(horizons.loc[4, "robust_effective"])


def test_analysis_fisher_randomization_finds_known_signal() -> None:
    generator = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=190)
    symbols = [f"S{index:02d}" for index in range(12)]
    latent = np.empty((len(dates), len(symbols)))
    latent[0] = generator.normal(size=len(symbols))
    for time_index in range(1, len(dates)):
        latent[time_index] = 0.75 * latent[time_index - 1] + generator.normal(
            scale=0.7, size=len(symbols)
        )
    signal = pd.DataFrame(latent, index=dates, columns=symbols)
    ranked = signal.rank(axis=1, pct=True).to_numpy()
    returns = np.zeros_like(latent)
    for time_index in range(2, len(dates)):
        rank_change = ranked[time_index - 1] - ranked[time_index - 2]
        returns[time_index] = (
            0.25 * returns[time_index - 1]
            + 0.035 * ranked[time_index - 1]
            + 0.12 * rank_change
            + generator.normal(scale=0.012, size=len(symbols))
        )
    close = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=symbols)
    noise_factor = pd.DataFrame(generator.normal(size=signal.shape), index=dates, columns=symbols)
    config = ScreeningConfig(
        train_sessions=126,
        test_sessions=21,
        forward_horizon=1,
        var_lags=2,
        permutations=99,
        q_threshold=0.20,
        granger_p_threshold=0.05,
        min_abs_rank_ic=0.0,
        min_sign_consistency=0.0,
        sensitivity_draws=0,
        min_sensitivity_rho=0.0,
        min_assets=8,
        min_dates=100,
        max_factors=2,
    )

    metrics = screen_factors(
        {"known_signal": signal, "noise": noise_factor},
        close,
        dates,
        config,
    ).set_index("factor")

    assert bool(metrics.loc["known_signal", "selected"])
    assert metrics.loc["known_signal", "mean_rank_ic"] > 0.15
    assert metrics.loc["known_signal", "granger_p_value"] < 0.05
    assert metrics.loc["known_signal", "fisher_q_value"] <= 0.20
    assert not bool(metrics.loc["noise", "selected"])
    assert metrics.loc["noise", "fisher_p_value"] > 0.05
    assert metrics.loc["known_signal", "causal_model"] == "reduced_rank_multivariate_var"
    assert metrics.loc["known_signal", "common_factor_count"] == 3
    assert metrics.loc["known_signal", "granger_restrictions"] > config.var_lags
    assert metrics.loc["known_signal", "cross_asset_signal_effect_norm"] > 0


def test_walk_forward_folds_have_embargo_and_no_label_overlap() -> None:
    index = pd.bdate_range("2020-01-02", periods=700)
    folds = make_walk_forward_folds(
        index,
        warmup_sessions=252,
        train_sessions=252,
        test_sessions=63,
        forward_horizon=5,
    )

    assert len(folds) == 3
    for fold in folds:
        assert len(fold.train_dates) == 252
        assert len(fold.embargo_dates) == 5
        assert len(fold.test_dates) == 63
        last_training_position = index.get_loc(fold.train_dates[-1])
        first_test_position = index.get_loc(fold.test_dates[0])
        assert last_training_position + 5 == first_test_position - 1


def _market_frames(periods: int = 650) -> dict[str, pd.DataFrame]:
    generator = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-02", periods=periods)
    symbols = [f"S{index:02d}" for index in range(8)] + ["SPY"]
    frames = {}
    for index, symbol in enumerate(symbols):
        returns = generator.normal(0.0003 + index * 0.00002, 0.012, periods)
        close = 100 * np.cumprod(1 + returns)
        open_price = close * (1 + generator.normal(0, 0.001, periods))
        frames[symbol] = pd.DataFrame(
            {
                "Date": dates,
                "Open": open_price,
                "High": np.maximum(open_price, close) * 1.005,
                "Low": np.minimum(open_price, close) * 0.995,
                "Close": close,
                "Volume": 1_000_000 + index * 100_000,
            }
        )
    return frames


def test_walk_forward_screen_runs_through_bt() -> None:
    frames = _market_frames()
    candidates = [symbol for symbol in frames if symbol != "SPY"]
    config = ScreeningConfig(
        train_sessions=252,
        test_sessions=63,
        forward_horizon=5,
        var_lags=2,
        permutations=19,
        q_threshold=1.0,
        granger_p_threshold=1.0,
        min_abs_rank_ic=0.0,
        min_sign_consistency=0.0,
        sensitivity_draws=0,
        min_sensitivity_rho=0.0,
        min_assets=6,
        min_dates=100,
        max_factors=2,
        top_n=2,
    )
    result = walk_forward_causal_screen(
        frames,
        candidates,
        factor_names=["momentum_21d", "reversal_5d"],
        config=config,
    )

    assert not result.screening_metrics.empty
    assert not result.horizon_metrics.empty
    assert set(result.horizon_metrics["horizon"]) == set(config.effect_horizons)
    assert len(result.folds) >= 4
    assert result.folds["selected_count"].ge(1).all()
    assert not result.target_weights.empty
    assert result.target_weights.abs().sum(axis=1).le(1.000001).all()
    assert {"causal_factor_composite", "buy_hold_SPY"}.issubset(
        result.backtest.equity_curve.columns
    )
    expected = forward_returns(
        pd.DataFrame({"SPY": frames["SPY"].set_index("Date")["Close"]}),
        5,
    )
    assert (
        expected.iloc[0, 0] == frames["SPY"]["Close"].iloc[5] / frames["SPY"]["Close"].iloc[0] - 1
    )
