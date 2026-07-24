from __future__ import annotations

import hashlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from .causal_screening import (
    _batch_panel_granger_f_statistics,
    _prepare_panel_granger_design,
)


@dataclass(frozen=True)
class BivariateSensitivityConfig:
    """Configuration for notebook-style, per-symbol sensitivity analysis."""

    var_lags: int = 7
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    sensitivity_draws: int = 999
    rho_limit: float = 0.99
    overturn_probability: float = 0.05
    kernel_bandwidth: float = 0.08
    stationarity_p_threshold: float = 0.05
    min_observations: int = 252
    workers: int = 0
    seed: int = 7

    def __post_init__(self) -> None:
        horizons = tuple(self.horizons)
        object.__setattr__(self, "horizons", horizons)
        if self.var_lags < 1:
            raise ValueError("var_lags must be positive")
        if not horizons or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons must be unique and increasing")
        if any(horizon < 1 for horizon in horizons):
            raise ValueError("horizons must be positive")
        if self.sensitivity_draws < 3 or self.sensitivity_draws % 2 == 0:
            raise ValueError("sensitivity_draws must be an odd integer of at least 3")
        if not 0 < self.rho_limit < 1:
            raise ValueError("rho_limit must be in (0, 1)")
        if not 0 < self.overturn_probability < 1:
            raise ValueError("overturn_probability must be in (0, 1)")
        if self.kernel_bandwidth <= 0:
            raise ValueError("kernel_bandwidth must be positive")
        if not 0 < self.stationarity_p_threshold < 1:
            raise ValueError("stationarity_p_threshold must be in (0, 1)")
        if self.min_observations <= 2 * self.var_lags + max(horizons):
            raise ValueError("min_observations is too small for the requested model")
        if self.workers < 0:
            raise ValueError("workers cannot be negative")


@dataclass(frozen=True)
class BivariateRhoResult:
    rho_star: pd.DataFrame
    exclusions: pd.DataFrame


@dataclass(frozen=True)
class _SymbolTask:
    symbol: str
    price: np.ndarray
    signal: np.ndarray
    config: BivariateSensitivityConfig


@dataclass(frozen=True)
class _SymbolResult:
    symbol: str
    rho_star: tuple[float, ...]
    error: str = ""


@dataclass(frozen=True)
class _BivariateVAR:
    intercept: np.ndarray
    coefficients: np.ndarray
    conditional_beta: float
    conditional_residuals: np.ndarray
    conditional_scale: float

    @property
    def lags(self) -> int:
        return self.coefficients.shape[0]


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def resolve_worker_count(requested: int, *, cpu_count: int | None = None) -> int:
    """Reserve eight CPUs in automatic mode and use one worker on small machines."""
    if requested < 0:
        raise ValueError("requested workers cannot be negative")
    if requested:
        return requested
    available = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, available - 8)


def _adf_p_value(values: np.ndarray, lags: int) -> float:
    if len(values) < 20 or float(np.std(values)) <= 1e-12:
        return 1.0
    maximum_lag = min(lags, max(len(values) // 2 - 2, 0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(adfuller(values, maxlag=maximum_lag, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        return 1.0


def _standardize(values: np.ndarray, label: str) -> np.ndarray:
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"{label} has no usable variation")
    return (values - float(np.mean(values))) / scale


def _prepare_series(task: _SymbolTask) -> tuple[np.ndarray, np.ndarray]:
    if task.price.shape != task.signal.shape or task.price.ndim != 1:
        raise ValueError("price and signal must be one-dimensional arrays of equal length")
    frame = pd.DataFrame({"price": task.price, "signal": task.signal})
    valid = frame.notna().all(axis=1).to_numpy()
    if not valid.any():
        raise ValueError("price and signal have no overlapping observations")
    first_valid = int(np.flatnonzero(valid)[0])
    frame = frame.iloc[first_valid:]
    if frame.isna().any(axis=None):
        raise ValueError("price or signal contains an internal missing observation")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("price and signal must be finite")
    if (frame["price"] <= 0).any():
        raise ValueError("backward total-return prices must be positive")

    outcome = np.diff(np.log(frame["price"].to_numpy(dtype=float)))
    signal = frame["signal"].to_numpy(dtype=float)[1:]
    if _adf_p_value(outcome, task.config.var_lags) > task.config.stationarity_p_threshold:
        raise ValueError("adjusted log returns are not stationary")
    if _adf_p_value(signal, task.config.var_lags) > task.config.stationarity_p_threshold:
        signal = np.diff(signal)
        outcome = outcome[1:]
        if _adf_p_value(signal, task.config.var_lags) > task.config.stationarity_p_threshold:
            raise ValueError("signal remains non-stationary after first differencing")
    if len(outcome) < task.config.min_observations:
        raise ValueError(
            f"only {len(outcome)} usable observations; need {task.config.min_observations}"
        )
    return _standardize(outcome, "outcome"), _standardize(signal, "signal")


def _fit_bivariate_var(outcome: np.ndarray, signal: np.ndarray, lags: int) -> _BivariateVAR:
    values = pd.DataFrame({"outcome": outcome, "signal": signal})
    fitted = VAR(values).fit(maxlags=lags, ic=None, trend="c")
    if fitted.k_ar != lags:
        raise ValueError("the bivariate VAR did not fit the requested lag order")
    if not fitted.is_stable(verbose=False):
        raise ValueError("the bivariate VAR is unstable")
    residuals = fitted.resid[["outcome", "signal"]].to_numpy(dtype=float)
    outcome_residual = residuals[:, 0]
    signal_residual = residuals[:, 1]
    outcome_variance = float(np.dot(outcome_residual, outcome_residual))
    if outcome_variance <= 1e-14:
        raise ValueError("the outcome VAR innovation has no usable variation")
    conditional_beta = float(np.dot(outcome_residual, signal_residual) / outcome_variance)
    conditional_residuals = signal_residual - conditional_beta * outcome_residual
    conditional_residuals -= conditional_residuals.mean()
    conditional_scale = float(np.sqrt(np.mean(conditional_residuals**2)))
    if conditional_scale <= 1e-12:
        raise ValueError("the conditional signal innovation has no usable variation")
    return _BivariateVAR(
        intercept=np.asarray(fitted.intercept, dtype=float),
        coefficients=np.asarray(fitted.coefs, dtype=float),
        conditional_beta=conditional_beta,
        conditional_residuals=conditional_residuals,
        conditional_scale=conditional_scale,
    )


def _outcome_confounder_errors(
    outcome: np.ndarray,
    *,
    lags: int,
    horizon: int,
) -> tuple[np.ndarray, float]:
    offsets = np.arange(horizon, horizon + lags)
    first_origin = int(offsets.max())
    origins = np.arange(first_origin, len(outcome))
    design = np.column_stack(
        [np.ones(len(origins)), *(outcome[origins - offset] for offset in offsets)]
    )
    coefficients = np.linalg.lstsq(design, outcome[origins], rcond=None)[0]
    residuals = outcome[origins] - design @ coefficients
    scale = float(np.sqrt(np.mean(residuals**2)))
    if scale <= 1e-12:
        raise ValueError("the outcome confounder model has no usable residual variation")
    aligned = np.zeros_like(outcome)
    aligned[origins] = residuals
    return aligned, scale


def _simulate_signals(
    model: _BivariateVAR,
    outcome: np.ndarray,
    signal: np.ndarray,
    rhos: np.ndarray,
    confounder_errors: np.ndarray,
    confounder_scale: float,
    generator: np.random.Generator,
) -> np.ndarray:
    draws = len(rhos)
    simulated = np.empty((draws, len(outcome)), dtype=float)
    simulated[:, : model.lags] = signal[None, : model.lags]
    residual_count = len(model.conditional_residuals)
    rho_noise_scale = np.sqrt(np.maximum(1.0 - rhos**2, 0.0))

    for time_index in range(model.lags, len(outcome)):
        outcome_prediction = np.full(draws, model.intercept[0], dtype=float)
        signal_prediction = np.full(draws, model.intercept[1], dtype=float)
        for lag in range(1, model.lags + 1):
            outcome_lag = outcome[time_index - lag]
            signal_lag = simulated[:, time_index - lag]
            coefficients = model.coefficients[lag - 1]
            outcome_prediction += coefficients[0, 0] * outcome_lag
            outcome_prediction += coefficients[0, 1] * signal_lag
            signal_prediction += coefficients[1, 0] * outcome_lag
            signal_prediction += coefficients[1, 1] * signal_lag
        outcome_error = outcome[time_index] - outcome_prediction
        conditional_mean = signal_prediction + model.conditional_beta * outcome_error
        sampled = model.conditional_residuals[generator.integers(residual_count, size=draws)]
        confounder = model.conditional_scale * confounder_errors[time_index] / confounder_scale
        innovation = rho_noise_scale * sampled + rhos * confounder
        simulated[:, time_index] = conditional_mean + innovation
    return simulated


def _smoothed_exceedance_probability(
    exceedances: np.ndarray,
    *,
    rho_limit: float,
    bandwidth: float,
) -> np.ndarray:
    spacing = 2.0 * rho_limit / max(len(exceedances) - 1, 1)
    sigma = max(bandwidth / spacing, 1.0)
    numerator = gaussian_filter1d(exceedances.astype(float), sigma=sigma, mode="constant", cval=0.0)
    denominator = gaussian_filter1d(
        np.ones(len(exceedances)), sigma=sigma, mode="constant", cval=0.0
    )
    return numerator / np.maximum(denominator, 1e-14)


def _rho_star_for_horizon(
    outcome: np.ndarray,
    signal: np.ndarray,
    model: _BivariateVAR,
    config: BivariateSensitivityConfig,
    *,
    horizon: int,
    seed: int,
) -> float:
    design = _prepare_panel_granger_design(outcome[:, None], lags=config.var_lags, horizon=horizon)
    observed = float(_batch_panel_granger_f_statistics(design, signal[:, None])[0])
    if not np.isfinite(observed):
        raise ValueError("the observed Granger statistic is not finite")
    confounder_errors, confounder_scale = _outcome_confounder_errors(
        outcome,
        lags=config.var_lags,
        horizon=horizon,
    )
    rhos = np.linspace(-config.rho_limit, config.rho_limit, config.sensitivity_draws)
    simulated = _simulate_signals(
        model,
        outcome,
        signal,
        rhos,
        confounder_errors,
        confounder_scale,
        np.random.default_rng(seed),
    )
    statistics = _batch_panel_granger_f_statistics(design, simulated[:, :, None])
    exceedances = np.isfinite(statistics) & (statistics >= observed)
    probability = _smoothed_exceedance_probability(
        exceedances,
        rho_limit=config.rho_limit,
        bandwidth=config.kernel_bandwidth,
    )
    overturned = np.flatnonzero(probability > config.overturn_probability)
    if len(overturned) == 0:
        return 1.0
    return float(np.min(np.abs(rhos[overturned])))


def _estimate_symbol(task: _SymbolTask) -> _SymbolResult:
    try:
        outcome, signal = _prepare_series(task)
        model = _fit_bivariate_var(outcome, signal, task.config.var_lags)
        rho_star = tuple(
            _rho_star_for_horizon(
                outcome,
                signal,
                model,
                task.config,
                horizon=horizon,
                seed=_stable_seed(task.config.seed, f"{task.symbol}:horizon-{horizon}"),
            )
            for horizon in task.config.horizons
        )
        return _SymbolResult(task.symbol, rho_star)
    except (ValueError, np.linalg.LinAlgError) as error:
        return _SymbolResult(
            task.symbol,
            tuple(np.nan for _ in task.config.horizons),
            str(error),
        )


def estimate_bivariate_rho(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    config: BivariateSensitivityConfig | None = None,
    *,
    analysis_dates: pd.DatetimeIndex | None = None,
) -> BivariateRhoResult:
    """Return one sensitivity tipping point for each symbol and forecast horizon."""
    config = config or BivariateSensitivityConfig()
    symbols = [symbol for symbol in prices.columns if symbol in signal.columns]
    if not symbols:
        raise ValueError("prices and signal have no common symbols")
    dates = (
        analysis_dates if analysis_dates is not None else prices.index.intersection(signal.index)
    )
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    if dates.empty:
        raise ValueError("analysis_dates is empty")
    tasks = [
        _SymbolTask(
            symbol=symbol,
            price=prices[symbol].reindex(dates).to_numpy(dtype=float),
            signal=signal[symbol].reindex(dates).to_numpy(dtype=float),
            config=config,
        )
        for symbol in symbols
    ]
    worker_count = min(resolve_worker_count(config.workers), len(tasks))
    if worker_count == 1:
        results = [_estimate_symbol(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_estimate_symbol, tasks, chunksize=1))

    columns = [f"rho_h{horizon}" for horizon in config.horizons]
    rho_star = pd.DataFrame(
        [result.rho_star for result in results],
        index=[result.symbol for result in results],
        columns=columns,
    )
    rho_star.index.name = "Symbol"
    exclusions = pd.DataFrame(
        [{"Symbol": result.symbol, "error": result.error} for result in results if result.error],
        columns=["Symbol", "error"],
    )
    return BivariateRhoResult(rho_star=rho_star, exclusions=exclusions)
