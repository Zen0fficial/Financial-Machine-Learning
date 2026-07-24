from __future__ import annotations

import hashlib
import warnings
from dataclasses import asdict, dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from statsmodels.tsa.stattools import adfuller

from .backtest import BacktestArtifacts, run_weight_backtest
from .data import close_matrix
from .factor_zoo import FACTOR_REGISTRY, compute_factor_zoo
from .factors import rank_target_weights, rebalance_dates


@dataclass(frozen=True)
class ScreeningConfig:
    train_sessions: int = 252
    test_sessions: int = 63
    forward_horizon: int = 5
    var_lags: int = 7
    common_factors: int = 3
    ridge_alpha: float = 1e-6
    covariance_shrinkage: float = 0.25
    permutations: int = 999
    simulation_batch_size: int = 8
    effect_horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    stationarity_p_threshold: float = 0.05
    stationarity_required_fraction: float = 0.80
    max_difference_order: int = 1
    q_threshold: float = 0.10
    granger_p_threshold: float = 0.05
    min_abs_rank_ic: float = 0.01
    min_sign_consistency: float = 0.52
    sensitivity_draws: int = 999
    min_sensitivity_rho: float = 0.10
    min_assets: int = 8
    min_dates: int = 126
    max_factors: int = 5
    top_n: int = 5
    rebalance_frequency: str = "weekly"
    long_short: bool = False
    seed: int = 7

    def __post_init__(self) -> None:
        horizons = tuple(self.effect_horizons)
        object.__setattr__(self, "effect_horizons", horizons)
        if self.train_sessions < 2 or self.test_sessions < 1:
            raise ValueError("train_sessions and test_sessions must be positive")
        if self.forward_horizon < 1 or self.var_lags < 1:
            raise ValueError("forward_horizon and var_lags must be positive")
        if self.common_factors < 0:
            raise ValueError("common_factors cannot be negative")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha cannot be negative")
        if not 0 <= self.covariance_shrinkage <= 1:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        if self.permutations < 1:
            raise ValueError("permutations must be positive")
        if self.simulation_batch_size < 1:
            raise ValueError("simulation_batch_size must be positive")
        if not horizons or any(horizon < 1 for horizon in horizons):
            raise ValueError("effect_horizons must contain positive integers")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("effect_horizons must be unique and increasing")
        if self.forward_horizon not in horizons:
            raise ValueError("forward_horizon must be included in effect_horizons")
        if not 0 < self.stationarity_p_threshold < 1:
            raise ValueError("stationarity_p_threshold must be in (0, 1)")
        if not 0 < self.stationarity_required_fraction <= 1:
            raise ValueError("stationarity_required_fraction must be in (0, 1]")
        if self.max_difference_order not in {0, 1}:
            raise ValueError("max_difference_order must be 0 or 1")
        if self.sensitivity_draws < 0:
            raise ValueError("sensitivity_draws cannot be negative")
        if 0 < self.sensitivity_draws < 3:
            raise ValueError("sensitivity_draws must be 0 or at least 3")
        if not 0 < self.q_threshold <= 1:
            raise ValueError("q_threshold must be in (0, 1]")
        if not 0 < self.granger_p_threshold <= 1:
            raise ValueError("granger_p_threshold must be in (0, 1]")
        if not 0 <= self.min_sign_consistency <= 1:
            raise ValueError("min_sign_consistency must be in [0, 1]")
        if not 0 <= self.min_sensitivity_rho <= 1:
            raise ValueError("min_sensitivity_rho must be in [0, 1]")
        if self.min_abs_rank_ic < 0:
            raise ValueError("min_abs_rank_ic cannot be negative")
        if self.min_assets < 3:
            raise ValueError("min_assets must be at least 3")
        if self.min_dates < 3:
            raise ValueError("min_dates must be at least 3")
        if self.max_factors < 1 or self.top_n < 1:
            raise ValueError("max_factors and top_n must be positive")


@dataclass(frozen=True)
class FoldDefinition:
    fold: int
    train_dates: pd.DatetimeIndex
    embargo_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


@dataclass
class WalkForwardResult:
    screening_metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame
    folds: pd.DataFrame
    selection_summary: pd.DataFrame
    target_weights: pd.DataFrame
    backtest: BacktestArtifacts
    run_config: dict[str, object]


@dataclass
class FactorScreeningResult:
    metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame


@dataclass(frozen=True)
class _StationaryPanel:
    values: pd.DataFrame
    difference_order: int
    level_stationary_fraction: float
    final_stationary_fraction: float


@dataclass(frozen=True)
class _PCAProjection:
    center: np.ndarray
    scale: np.ndarray
    loadings: np.ndarray

    @property
    def factor_count(self) -> int:
        return self.loadings.shape[1]

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = (values - self.center) / self.scale
        return standardized @ self.loadings


@dataclass(frozen=True)
class _MultivariateVAR:
    outcome_projection: _PCAProjection
    signal_projection: _PCAProjection
    fixed_effects: np.ndarray
    outcome_own_lags: np.ndarray
    signal_own_lags: np.ndarray
    outcome_to_outcome_lags: np.ndarray
    signal_to_outcome_lags: np.ndarray
    outcome_to_signal_lags: np.ndarray
    signal_to_signal_lags: np.ndarray
    conditional_beta: np.ndarray
    conditional_residuals: np.ndarray
    conditional_scale: np.ndarray
    signal_r_squared: float

    @property
    def lags(self) -> int:
        return self.outcome_own_lags.shape[0]


@dataclass(frozen=True)
class _PanelGrangerDesign:
    target: np.ndarray
    restricted: np.ndarray
    restricted_cross_inverse: np.ndarray
    restricted_target_cross: np.ndarray
    restricted_ssr: float
    origins: np.ndarray
    lags: int
    horizon: int
    asset_count: int
    degrees_denominator: int


@dataclass(frozen=True)
class _MultivariateGrangerDesign:
    target: np.ndarray
    restricted: np.ndarray
    restricted_cross_inverse: np.ndarray
    restricted_target_cross: np.ndarray
    restricted_ssr: float
    origins: np.ndarray
    lags: int
    horizon: int
    asset_count: int
    degrees_numerator: int
    degrees_denominator: int
    outcome_projection: _PCAProjection
    signal_projection: _PCAProjection


@dataclass(frozen=True)
class _FactorAnalysis:
    outcome: np.ndarray
    signal: np.ndarray
    model: _MultivariateVAR
    granger_design: _MultivariateGrangerDesign
    granger_f: float
    granger_p: float
    panel_observations: int
    outcome_difference_order: int
    signal_difference_order: int
    outcome_level_stationary_fraction: float
    signal_level_stationary_fraction: float
    outcome_final_stationary_fraction: float
    signal_final_stationary_fraction: float


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return close.shift(-horizon) / close - 1.0


def cross_sectional_rank(values: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranks centered at zero, calculated independently by date."""
    return values.rank(axis=1, method="average", pct=True) - 0.5


def _rowwise_correlation(
    left: np.ndarray,
    right: np.ndarray,
    min_assets: int,
) -> np.ndarray:
    mask = np.isfinite(left) & np.isfinite(right)
    count = mask.sum(axis=1)
    left_sum = np.where(mask, left, 0.0).sum(axis=1)
    right_sum = np.where(mask, right, 0.0).sum(axis=1)
    left_mean = np.divide(left_sum, count, out=np.zeros_like(left_sum), where=count > 0)
    right_mean = np.divide(right_sum, count, out=np.zeros_like(right_sum), where=count > 0)
    left_centered = np.where(mask, left - left_mean[:, None], 0.0)
    right_centered = np.where(mask, right - right_mean[:, None], 0.0)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1))
    output = np.full(left.shape[0], np.nan)
    valid = (count >= min_assets) & (denominator > 1e-14)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def daily_rank_ic(
    factor: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    min_assets: int,
) -> pd.Series:
    values = _rowwise_correlation(
        cross_sectional_rank(factor).to_numpy(dtype=float),
        cross_sectional_rank(outcome).to_numpy(dtype=float),
        min_assets,
    )
    return pd.Series(values, index=factor.index, name="rank_ic").dropna()


def newey_west_mean_test(values: pd.Series, max_lags: int) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    size = len(clean)
    if size < 3:
        return np.nan, np.nan, 1.0
    mean = float(clean.mean())
    errors = clean - mean
    lag_count = min(max_lags, size - 1)
    long_run_variance = float(np.dot(errors, errors) / size)
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        covariance = float(np.dot(errors[lag:], errors[:-lag]) / size)
        long_run_variance += 2.0 * weight * covariance
    standard_error = sqrt(max(long_run_variance, 0.0) / size)
    if standard_error <= 1e-14:
        return mean, np.nan, 1.0
    statistic = mean / standard_error
    return mean, statistic, erfc(abs(statistic) / sqrt(2.0))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, preserving the input index."""
    clean = p_values.fillna(1.0).clip(0.0, 1.0)
    order = np.argsort(clean.to_numpy())
    ranked = clean.to_numpy()[order]
    count = len(ranked)
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    output = np.empty(count, dtype=float)
    output[order] = adjusted
    return pd.Series(output, index=p_values.index, name="fisher_q_value")


def _fixed_effect_design(asset_count: int, row_count: int) -> np.ndarray:
    return np.tile(np.eye(asset_count), (row_count, 1))


def _fit_pca_projection(values: np.ndarray, factor_count: int) -> _PCAProjection:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("PCA input must be a finite time-by-asset matrix")
    time_count, asset_count = values.shape
    usable_factors = min(factor_count, max(asset_count - 1, 0), max(time_count - 1, 0))
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    if usable_factors == 0:
        loadings = np.empty((asset_count, 0), dtype=float)
    else:
        standardized = (values - center) / scale
        _, _, right_vectors = np.linalg.svd(standardized, full_matrices=False)
        loadings = right_vectors[:usable_factors].T.copy()
        for column in range(loadings.shape[1]):
            anchor = int(np.argmax(np.abs(loadings[:, column])))
            if loadings[anchor, column] < 0:
                loadings[:, column] *= -1.0
    return _PCAProjection(center=center, scale=scale, loadings=loadings)


def _factor_interactions(scores: np.ndarray, target_loadings: np.ndarray) -> np.ndarray:
    """Create low-rank cross-asset regressors in time-major observation order."""
    if scores.ndim != 2:
        raise ValueError("factor scores must be two-dimensional")
    time_count, predictor_factors = scores.shape
    asset_count, target_factors = target_loadings.shape
    if predictor_factors == 0 or target_factors == 0:
        return np.empty((time_count * asset_count, 0), dtype=float)
    interactions = np.einsum("tj,ik->tikj", scores, target_loadings, optimize=True)
    return interactions.reshape(time_count * asset_count, target_factors * predictor_factors)


def _batch_factor_interactions(
    scores: np.ndarray,
    target_loadings: np.ndarray,
) -> np.ndarray:
    if scores.ndim != 3:
        raise ValueError("batched factor scores must be three-dimensional")
    draws, time_count, predictor_factors = scores.shape
    asset_count, target_factors = target_loadings.shape
    if predictor_factors == 0 or target_factors == 0:
        return np.empty((draws, time_count * asset_count, 0), dtype=float)
    interactions = np.einsum("btj,ik->btikj", scores, target_loadings, optimize=True)
    return interactions.reshape(
        draws,
        time_count * asset_count,
        target_factors * predictor_factors,
    )


def _prepare_panel_granger_design(
    outcome: np.ndarray,
    *,
    lags: int,
    horizon: int,
) -> _PanelGrangerDesign:
    time_count, asset_count = outcome.shape
    final_origin = time_count - horizon
    if final_origin < lags:
        raise ValueError("not enough rows for the Granger regression")

    origins = np.arange(lags, final_origin + 1)
    target = outcome[origins + horizon - 1].reshape(-1)
    outcome_predictors = np.column_stack(
        [outcome[origins + horizon - 1 - lag].reshape(-1) for lag in range(1, lags + 1)]
    )
    fixed_effects = _fixed_effect_design(asset_count, len(origins))
    restricted = np.column_stack([fixed_effects, outcome_predictors])
    valid = np.isfinite(target) & np.isfinite(restricted).all(axis=1)
    target = target[valid]
    restricted = restricted[valid]
    observations = len(target)
    degrees_denominator = observations - restricted.shape[1] - lags
    if degrees_denominator <= 0 or observations == 0:
        raise ValueError("not enough panel observations for the Granger regression")

    restricted_cross_inverse = np.linalg.pinv(restricted.T @ restricted)
    restricted_target_cross = restricted.T @ target
    restricted_coefficients = restricted_cross_inverse @ restricted_target_cross
    restricted_residual = target - restricted @ restricted_coefficients
    return _PanelGrangerDesign(
        target=target,
        restricted=restricted,
        restricted_cross_inverse=restricted_cross_inverse,
        restricted_target_cross=restricted_target_cross,
        restricted_ssr=float(np.dot(restricted_residual, restricted_residual)),
        origins=origins,
        lags=lags,
        horizon=horizon,
        asset_count=asset_count,
        degrees_denominator=degrees_denominator,
    )


def _batch_panel_granger_f_statistics(
    design: _PanelGrangerDesign,
    signals: np.ndarray,
) -> np.ndarray:
    """Evaluate Fisher draws using the Frisch-Waugh-Lovell form of the F-test."""
    if signals.ndim == 2:
        signals = signals[None, ...]
    signal_predictors = np.stack(
        [
            signals[:, design.origins - lag, :].reshape(len(signals), -1)
            for lag in range(1, design.lags + 1)
        ],
        axis=2,
    )
    if not np.isfinite(signal_predictors).all():
        return np.full(len(signals), np.nan)

    restricted_signal_cross = np.einsum(
        "nk,bnp->bkp", design.restricted, signal_predictors, optimize=True
    )
    signal_cross = np.einsum("bnp,bnq->bpq", signal_predictors, signal_predictors, optimize=True)
    signal_target_cross = np.einsum("bnp,n->bp", signal_predictors, design.target, optimize=True)
    residualized_signal_cross = signal_cross - np.einsum(
        "bpk,kl,blq->bpq",
        restricted_signal_cross.transpose(0, 2, 1),
        design.restricted_cross_inverse,
        restricted_signal_cross,
        optimize=True,
    )
    residualized_signal_target = signal_target_cross - np.einsum(
        "bpk,kl,l->bp",
        restricted_signal_cross.transpose(0, 2, 1),
        design.restricted_cross_inverse,
        design.restricted_target_cross,
        optimize=True,
    )

    coefficients = np.empty_like(residualized_signal_target)
    for draw in range(len(signals)):
        coefficients[draw] = np.linalg.lstsq(
            residualized_signal_cross[draw],
            residualized_signal_target[draw],
            rcond=None,
        )[0]
    reduction = np.einsum("bp,bp->b", residualized_signal_target, coefficients, optimize=True)
    reduction = np.clip(reduction, 0.0, design.restricted_ssr)
    unrestricted_ssr = design.restricted_ssr - reduction
    statistics = np.full(len(signals), np.inf)
    positive = unrestricted_ssr > 1e-14
    statistics[positive] = (reduction[positive] / design.lags) / (
        unrestricted_ssr[positive] / design.degrees_denominator
    )
    return statistics


def _panel_granger_f_statistic(
    outcome: np.ndarray,
    signal: np.ndarray,
    *,
    lags: int,
    horizon: int,
) -> tuple[float, float, int]:
    """Pooled version of the notebook's Granger SSR F-test.

    The restrictions and time shift match ``grangercausalitytests``. Stock fixed
    effects adapt the single-series statistic to a factor rule applied across a
    stock universe.
    """
    try:
        design = _prepare_panel_granger_design(outcome, lags=lags, horizon=horizon)
    except ValueError:
        return np.nan, 1.0, 0
    statistic = float(_batch_panel_granger_f_statistics(design, signal)[0])
    p_value = float(f_distribution.sf(statistic, design.lags, design.degrees_denominator))
    return statistic, p_value, len(design.target)


def _prepare_multivariate_granger_design(
    outcome: np.ndarray,
    signal: np.ndarray,
    *,
    lags: int,
    horizon: int,
    common_factors: int,
) -> _MultivariateGrangerDesign:
    if outcome.shape != signal.shape or outcome.ndim != 2:
        raise ValueError("outcome and signal must have the same time-by-asset shape")
    if not np.isfinite(outcome).all() or not np.isfinite(signal).all():
        raise ValueError("multivariate Granger inputs must be finite")
    time_count, asset_count = outcome.shape
    final_origin = time_count - horizon
    if final_origin < lags:
        raise ValueError("not enough rows for the multivariate Granger regression")

    outcome_projection = _fit_pca_projection(outcome, common_factors)
    signal_projection = _fit_pca_projection(signal, common_factors)
    outcome_scores = outcome_projection.transform(outcome)
    origins = np.arange(lags, final_origin + 1)
    target = outcome[origins + horizon - 1].reshape(-1)
    restricted_blocks = [_fixed_effect_design(asset_count, len(origins))]
    for lag in range(1, lags + 1):
        outcome_indices = origins - lag
        restricted_blocks.append(outcome[outcome_indices].reshape(-1, 1))
        restricted_blocks.append(
            _factor_interactions(
                outcome_scores[outcome_indices],
                outcome_projection.loadings,
            )
        )
    restricted = np.column_stack(restricted_blocks)
    factor_restrictions = (
        outcome_projection.factor_count * signal_projection.factor_count
    )
    degrees_numerator = lags * (1 + factor_restrictions)
    observations = len(target)
    degrees_denominator = observations - restricted.shape[1] - degrees_numerator
    if degrees_denominator <= 0:
        raise ValueError("not enough observations for the multivariate Granger regression")

    restricted_cross_inverse = np.linalg.pinv(restricted.T @ restricted)
    restricted_target_cross = restricted.T @ target
    restricted_coefficients = restricted_cross_inverse @ restricted_target_cross
    restricted_residual = target - restricted @ restricted_coefficients
    return _MultivariateGrangerDesign(
        target=target,
        restricted=restricted,
        restricted_cross_inverse=restricted_cross_inverse,
        restricted_target_cross=restricted_target_cross,
        restricted_ssr=float(np.dot(restricted_residual, restricted_residual)),
        origins=origins,
        lags=lags,
        horizon=horizon,
        asset_count=asset_count,
        degrees_numerator=degrees_numerator,
        degrees_denominator=degrees_denominator,
        outcome_projection=outcome_projection,
        signal_projection=signal_projection,
    )


def _batch_multivariate_granger_f_statistics(
    design: _MultivariateGrangerDesign,
    signals: np.ndarray,
) -> np.ndarray:
    """Test direct and reduced-rank cross-asset signal restrictions jointly."""
    if signals.ndim == 2:
        signals = signals[None, ...]
    if signals.ndim != 3 or signals.shape[2] != design.asset_count:
        raise ValueError("signals must be draw-by-time-by-asset")
    signal_scores = design.signal_projection.transform(signals)
    predictor_blocks: list[np.ndarray] = []
    for lag in range(1, design.lags + 1):
        signal_indices = design.origins - lag
        predictor_blocks.append(
            signals[:, signal_indices].reshape(len(signals), -1, 1)
        )
        predictor_blocks.append(
            _batch_factor_interactions(
                signal_scores[:, signal_indices],
                design.outcome_projection.loadings,
            )
        )
    signal_predictors = np.concatenate(predictor_blocks, axis=2)
    if not np.isfinite(signal_predictors).all():
        return np.full(len(signals), np.nan)

    restricted_signal_cross = np.einsum(
        "nk,bnp->bkp", design.restricted, signal_predictors, optimize=True
    )
    signal_cross = np.einsum("bnp,bnq->bpq", signal_predictors, signal_predictors, optimize=True)
    signal_target_cross = np.einsum("bnp,n->bp", signal_predictors, design.target, optimize=True)
    residualized_signal_cross = signal_cross - np.einsum(
        "bpk,kl,blq->bpq",
        restricted_signal_cross.transpose(0, 2, 1),
        design.restricted_cross_inverse,
        restricted_signal_cross,
        optimize=True,
    )
    residualized_signal_target = signal_target_cross - np.einsum(
        "bpk,kl,l->bp",
        restricted_signal_cross.transpose(0, 2, 1),
        design.restricted_cross_inverse,
        design.restricted_target_cross,
        optimize=True,
    )

    coefficients = np.empty_like(residualized_signal_target)
    for draw in range(len(signals)):
        coefficients[draw] = np.linalg.lstsq(
            residualized_signal_cross[draw],
            residualized_signal_target[draw],
            rcond=None,
        )[0]
    reduction = np.einsum("bp,bp->b", residualized_signal_target, coefficients, optimize=True)
    reduction = np.clip(reduction, 0.0, design.restricted_ssr)
    unrestricted_ssr = design.restricted_ssr - reduction
    statistics = np.full(len(signals), np.inf)
    positive = unrestricted_ssr > 1e-14
    statistics[positive] = (reduction[positive] / design.degrees_numerator) / (
        unrestricted_ssr[positive] / design.degrees_denominator
    )
    return statistics


def _multivariate_granger_f_statistic(
    outcome: np.ndarray,
    signal: np.ndarray,
    *,
    lags: int,
    horizon: int,
    common_factors: int,
) -> tuple[float, float, int]:
    try:
        design = _prepare_multivariate_granger_design(
            outcome,
            signal,
            lags=lags,
            horizon=horizon,
            common_factors=common_factors,
        )
    except ValueError:
        return np.nan, 1.0, 0
    statistic = float(_batch_multivariate_granger_f_statistics(design, signal)[0])
    p_value = float(
        f_distribution.sf(
            statistic,
            design.degrees_numerator,
            design.degrees_denominator,
        )
    )
    return statistic, p_value, len(design.target)


def _multivariate_equation_design(
    outcome: np.ndarray,
    signal: np.ndarray,
    outcome_scores: np.ndarray,
    signal_scores: np.ndarray,
    target_loadings: np.ndarray,
    *,
    lags: int,
) -> np.ndarray:
    time_count, asset_count = outcome.shape
    origins = np.arange(lags, time_count)
    blocks = [_fixed_effect_design(asset_count, len(origins))]
    for lag in range(1, lags + 1):
        indices = origins - lag
        blocks.extend(
            [
                outcome[indices].reshape(-1, 1),
                signal[indices].reshape(-1, 1),
                _factor_interactions(outcome_scores[indices], target_loadings),
                _factor_interactions(signal_scores[indices], target_loadings),
            ]
        )
    return np.column_stack(blocks)


def _ridge_fit(design: np.ndarray, target: np.ndarray, fixed_effects: int, alpha: float) -> np.ndarray:
    cross = design.T @ design
    target_cross = design.T @ target
    if alpha > 0:
        penalty = np.zeros_like(cross)
        diagonal = np.diag(cross)
        penalty_values = alpha * np.maximum(diagonal[fixed_effects:], 1e-12)
        penalty[fixed_effects:, fixed_effects:] = np.diag(penalty_values)
        cross = cross + penalty
    return np.linalg.pinv(cross) @ target_cross


def _parse_multivariate_coefficients(
    coefficients: np.ndarray,
    *,
    asset_count: int,
    lags: int,
    target_factors: int,
    outcome_factors: int,
    signal_factors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = asset_count
    own_outcome = np.empty(lags, dtype=float)
    own_signal = np.empty(lags, dtype=float)
    outcome_factor = np.empty((lags, target_factors, outcome_factors), dtype=float)
    signal_factor = np.empty((lags, target_factors, signal_factors), dtype=float)
    for lag in range(lags):
        own_outcome[lag] = coefficients[position]
        position += 1
        own_signal[lag] = coefficients[position]
        position += 1
        outcome_size = target_factors * outcome_factors
        outcome_factor[lag] = coefficients[position : position + outcome_size].reshape(
            target_factors, outcome_factors
        )
        position += outcome_size
        signal_size = target_factors * signal_factors
        signal_factor[lag] = coefficients[position : position + signal_size].reshape(
            target_factors, signal_factors
        )
        position += signal_size
    if position != len(coefficients):
        raise ValueError("unexpected multivariate coefficient count")
    return (
        coefficients[:asset_count],
        own_outcome,
        own_signal,
        outcome_factor,
        signal_factor,
    )


def _fit_multivariate_var(
    outcome: np.ndarray,
    signal: np.ndarray,
    *,
    lags: int,
    common_factors: int,
    ridge_alpha: float,
    covariance_shrinkage: float,
) -> _MultivariateVAR:
    if outcome.shape != signal.shape or outcome.ndim != 2:
        raise ValueError("outcome and signal must have the same time-by-asset shape")
    if not np.isfinite(outcome).all() or not np.isfinite(signal).all():
        raise ValueError("multivariate VAR inputs must be finite")
    time_count, asset_count = outcome.shape
    if time_count <= lags:
        raise ValueError("not enough rows to fit the multivariate VAR")

    outcome_projection = _fit_pca_projection(outcome, common_factors)
    signal_projection = _fit_pca_projection(signal, common_factors)
    outcome_scores = outcome_projection.transform(outcome)
    signal_scores = signal_projection.transform(signal)
    origins = np.arange(lags, time_count)
    equation_results = []
    residual_panels = []

    for response, target_projection in (
        (outcome, outcome_projection),
        (signal, signal_projection),
    ):
        design = _multivariate_equation_design(
            outcome,
            signal,
            outcome_scores,
            signal_scores,
            target_projection.loadings,
            lags=lags,
        )
        target = response[origins].reshape(-1)
        if len(target) <= design.shape[1]:
            raise ValueError("not enough observations to fit the multivariate VAR")
        coefficients = _ridge_fit(design, target, asset_count, ridge_alpha)
        residual_panels.append((target - design @ coefficients).reshape(len(origins), asset_count))
        equation_results.append(
            _parse_multivariate_coefficients(
                coefficients,
                asset_count=asset_count,
                lags=lags,
                target_factors=target_projection.factor_count,
                outcome_factors=outcome_projection.factor_count,
                signal_factors=signal_projection.factor_count,
            )
        )

    outcome_result, signal_result = equation_results
    outcome_residual, signal_residual = residual_panels
    centered_outcome_residual = outcome_residual - outcome_residual.mean(axis=0)
    centered_signal_residual = signal_residual - signal_residual.mean(axis=0)
    residual_rows = len(centered_outcome_residual)
    covariance_yy = centered_outcome_residual.T @ centered_outcome_residual / residual_rows
    covariance_sy = centered_signal_residual.T @ centered_outcome_residual / residual_rows
    diagonal_covariance = np.diag(np.diag(covariance_yy))
    shrunk_covariance = (
        (1.0 - covariance_shrinkage) * covariance_yy
        + covariance_shrinkage * diagonal_covariance
    )
    variance_floor = max(float(np.trace(covariance_yy)) / max(asset_count, 1) * 1e-8, 1e-14)
    shrunk_covariance = shrunk_covariance + np.eye(asset_count) * variance_floor
    conditional_beta = covariance_sy @ np.linalg.pinv(shrunk_covariance)
    conditional_residuals = centered_signal_residual - (
        centered_outcome_residual @ conditional_beta.T
    )
    conditional_residuals -= conditional_residuals.mean(axis=0)
    conditional_scale = np.sqrt(np.mean(conditional_residuals**2, axis=0))
    if not np.isfinite(conditional_residuals).all() or np.max(conditional_scale) <= 1e-14:
        raise ValueError("the multivariate VAR has no usable conditional signal residuals")

    signal_target = signal[origins]
    total_variance = float(np.var(signal_target))
    signal_r_squared = (
        1.0 - float(np.var(signal_residual)) / total_variance if total_variance > 1e-14 else np.nan
    )
    return _MultivariateVAR(
        outcome_projection=outcome_projection,
        signal_projection=signal_projection,
        fixed_effects=np.column_stack([outcome_result[0], signal_result[0]]),
        outcome_own_lags=np.column_stack([outcome_result[1], signal_result[1]]),
        signal_own_lags=np.column_stack([outcome_result[2], signal_result[2]]),
        outcome_to_outcome_lags=outcome_result[3],
        signal_to_outcome_lags=outcome_result[4],
        outcome_to_signal_lags=signal_result[3],
        signal_to_signal_lags=signal_result[4],
        conditional_beta=conditional_beta,
        conditional_residuals=conditional_residuals,
        conditional_scale=conditional_scale,
        signal_r_squared=signal_r_squared,
    )


def _simulate_conditional_signals(
    model: _MultivariateVAR,
    outcome: np.ndarray,
    initial_signal: np.ndarray,
    generator: np.random.Generator,
    *,
    draws: int,
    rhos: np.ndarray | None = None,
    confounder_errors: np.ndarray | None = None,
    confounder_scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Condition a reduced-rank multivariate signal process on realized returns."""
    if draws < 1:
        raise ValueError("draws must be positive")
    if rhos is None:
        rhos = np.zeros(draws)
    rhos = np.asarray(rhos, dtype=float)
    if rhos.shape != (draws,) or (np.abs(rhos) > 1).any():
        raise ValueError("rhos must contain one value in [-1, 1] per draw")
    time_count, asset_count = outcome.shape
    simulated = np.empty((draws, time_count, asset_count), dtype=float)
    simulated[:, : model.lags] = initial_signal[None, : model.lags]
    residual_rows = len(model.conditional_residuals)
    outcome_loadings = model.outcome_projection.loadings
    signal_loadings = model.signal_projection.loadings

    for time_index in range(model.lags, time_count):
        prediction = np.broadcast_to(model.fixed_effects, (draws, asset_count, 2)).copy()
        for lag in range(1, model.lags + 1):
            outcome_lag = outcome[time_index - lag]
            signal_lag = simulated[:, time_index - lag]
            prediction += (
                outcome_lag[None, :, None] * model.outcome_own_lags[lag - 1]
            )
            prediction += signal_lag[:, :, None] * model.signal_own_lags[lag - 1]

            outcome_scores = model.outcome_projection.transform(outcome_lag)
            signal_scores = model.signal_projection.transform(signal_lag)
            if model.outcome_projection.factor_count:
                prediction[:, :, 0] += np.einsum(
                    "ik,kj,j->i",
                    outcome_loadings,
                    model.outcome_to_outcome_lags[lag - 1],
                    outcome_scores,
                    optimize=True,
                )[None]
                prediction[:, :, 1] += np.einsum(
                    "ik,kj,j->i",
                    signal_loadings,
                    model.outcome_to_signal_lags[lag - 1],
                    outcome_scores,
                    optimize=True,
                )[None]
            if model.signal_projection.factor_count:
                prediction[:, :, 0] += np.einsum(
                    "ik,kj,bj->bi",
                    outcome_loadings,
                    model.signal_to_outcome_lags[lag - 1],
                    signal_scores,
                    optimize=True,
                )
                prediction[:, :, 1] += np.einsum(
                    "ik,kj,bj->bi",
                    signal_loadings,
                    model.signal_to_signal_lags[lag - 1],
                    signal_scores,
                    optimize=True,
                )
        outcome_error = outcome[None, time_index] - prediction[:, :, 0]
        conditional_mean = prediction[:, :, 1] + outcome_error @ model.conditional_beta.T
        sampled = model.conditional_residuals[generator.integers(residual_rows, size=draws)]
        innovation = sampled
        if confounder_errors is not None:
            scale = np.asarray(confounder_scale, dtype=float)
            scaled_confounder = (
                model.conditional_scale[None]
                * confounder_errors[None, time_index]
                / np.maximum(scale, 1e-14)
            )
            innovation = (
                np.sqrt(np.maximum(1.0 - rhos**2, 0.0))[:, None] * sampled
                + rhos[:, None] * scaled_confounder
            )
        simulated[:, time_index] = conditional_mean + innovation
    return simulated


def _adf_stationary_fraction(values: pd.DataFrame, config: ScreeningConfig) -> float:
    stationary = 0
    tested = 0
    for column in values.columns:
        series = values[column].dropna().to_numpy(dtype=float)
        if len(series) < max(20, 2 * config.var_lags + 5):
            continue
        tested += 1
        if float(np.std(series)) <= 1e-12:
            stationary += 1
            continue
        maximum_lag = min(config.var_lags, max(len(series) // 2 - 2, 0))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_value = float(adfuller(series, maxlag=maximum_lag, autolag="AIC")[1])
        except (ValueError, np.linalg.LinAlgError):
            p_value = 1.0
        stationary += int(p_value <= config.stationarity_p_threshold)
    return stationary / tested if tested else 0.0


def _stationarize_panel(
    values: pd.DataFrame,
    config: ScreeningConfig,
    *,
    label: str,
    force_difference: bool = False,
) -> _StationaryPanel:
    level_fraction = _adf_stationary_fraction(values, config)
    difference_order = 0
    transformed = values.copy()
    final_fraction = level_fraction
    if force_difference or level_fraction < config.stationarity_required_fraction:
        if config.max_difference_order == 0 and not force_difference:
            raise ValueError(f"{label} panel is non-stationary and differencing is disabled")
        difference_order = 1
        transformed = values.diff()
        final_fraction = _adf_stationary_fraction(transformed, config)
    if final_fraction < config.stationarity_required_fraction:
        raise ValueError(
            f"{label} panel remains non-stationary after difference order {difference_order}"
        )
    return _StationaryPanel(
        values=transformed,
        difference_order=difference_order,
        level_stationary_fraction=level_fraction,
        final_stationary_fraction=final_fraction,
    )


def _stationary_factor_panel(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: ScreeningConfig,
    outcome_panel: _StationaryPanel | None = None,
) -> tuple[np.ndarray, np.ndarray, _StationaryPanel, _StationaryPanel]:
    ranked_signal = cross_sectional_rank(factor).reindex(dates)
    if outcome_panel is None:
        log_price = np.log(close.reindex(dates))
        # Outcome is always log returns: US log-price levels are integrated and
        # ADF on ~252 observations has low power to detect the unit root, so the
        # level-vs-difference decision is fixed at first-differencing here.
        outcome_panel = _stationarize_panel(
            log_price, config, label="log-price", force_difference=True
        )
    signal_panel = _stationarize_panel(ranked_signal, config, label="factor-rank")
    stationary = pd.concat(
        {"outcome": outcome_panel.values, "signal": signal_panel.values},
        axis=1,
    )
    valid_dates = stationary["outcome"].notna().all(axis=1) & stationary["signal"].notna().all(
        axis=1
    )
    stationary = stationary.loc[valid_dates]
    return (
        stationary["outcome"].to_numpy(dtype=float),
        stationary["signal"].to_numpy(dtype=float),
        outcome_panel,
        signal_panel,
    )


def _analyze_factor(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: ScreeningConfig,
    outcome_panel: _StationaryPanel | None = None,
) -> _FactorAnalysis:
    outcome, signal, outcome_panel, signal_panel = _stationary_factor_panel(
        factor,
        close,
        dates,
        config,
        outcome_panel,
    )
    model = _fit_multivariate_var(
        outcome,
        signal,
        lags=config.var_lags,
        common_factors=config.common_factors,
        ridge_alpha=config.ridge_alpha,
        covariance_shrinkage=config.covariance_shrinkage,
    )
    granger_design = _prepare_multivariate_granger_design(
        outcome,
        signal,
        lags=config.var_lags,
        horizon=config.forward_horizon,
        common_factors=config.common_factors,
    )
    granger_f = float(_batch_multivariate_granger_f_statistics(granger_design, signal)[0])
    granger_p = float(
        f_distribution.sf(
            granger_f,
            granger_design.degrees_numerator,
            granger_design.degrees_denominator,
        )
    )
    return _FactorAnalysis(
        outcome=outcome,
        signal=signal,
        model=model,
        granger_design=granger_design,
        granger_f=granger_f,
        granger_p=granger_p,
        panel_observations=len(granger_design.target),
        outcome_difference_order=outcome_panel.difference_order,
        signal_difference_order=signal_panel.difference_order,
        outcome_level_stationary_fraction=outcome_panel.level_stationary_fraction,
        signal_level_stationary_fraction=signal_panel.level_stationary_fraction,
        outcome_final_stationary_fraction=outcome_panel.final_stationary_fraction,
        signal_final_stationary_fraction=signal_panel.final_stationary_fraction,
    )


def _analysis_for_horizon(
    base: _FactorAnalysis,
    horizon: int,
    config: ScreeningConfig,
) -> _FactorAnalysis:
    if horizon == base.granger_design.horizon:
        return base
    granger_design = _prepare_multivariate_granger_design(
        base.outcome,
        base.signal,
        lags=config.var_lags,
        horizon=horizon,
        common_factors=config.common_factors,
    )
    granger_f = float(_batch_multivariate_granger_f_statistics(granger_design, base.signal)[0])
    granger_p = float(
        f_distribution.sf(
            granger_f,
            granger_design.degrees_numerator,
            granger_design.degrees_denominator,
        )
    )
    return _FactorAnalysis(
        outcome=base.outcome,
        signal=base.signal,
        model=base.model,
        granger_design=granger_design,
        granger_f=granger_f,
        granger_p=granger_p,
        panel_observations=len(granger_design.target),
        outcome_difference_order=base.outcome_difference_order,
        signal_difference_order=base.signal_difference_order,
        outcome_level_stationary_fraction=base.outcome_level_stationary_fraction,
        signal_level_stationary_fraction=base.signal_level_stationary_fraction,
        outcome_final_stationary_fraction=base.outcome_final_stationary_fraction,
        signal_final_stationary_fraction=base.signal_final_stationary_fraction,
    )


def _fisher_randomization_p_value(
    analysis: _FactorAnalysis,
    config: ScreeningConfig,
    generator: np.random.Generator,
) -> float:
    if not np.isfinite(analysis.granger_f):
        return 1.0
    statistics = _randomized_multivariate_statistics(
        analysis,
        generator,
        draws=config.permutations,
        batch_size=config.simulation_batch_size,
    )
    exceedances = int(np.sum(np.isfinite(statistics) & (statistics >= analysis.granger_f)))
    return (exceedances + 1.0) / (config.permutations + 1.0)


def _fisher_horizon_p_values(
    analyses: dict[int, _FactorAnalysis],
    config: ScreeningConfig,
    generator: np.random.Generator,
) -> dict[int, float]:
    if not analyses:
        return {}
    base = next(iter(analyses.values()))
    exceedances = dict.fromkeys(analyses, 0)
    for start in range(0, config.permutations, config.simulation_batch_size):
        stop = min(start + config.simulation_batch_size, config.permutations)
        randomized_signals = _simulate_conditional_signals(
            base.model,
            base.outcome,
            base.signal,
            generator,
            draws=stop - start,
        )
        for horizon, analysis in analyses.items():
            statistics = _batch_multivariate_granger_f_statistics(
                analysis.granger_design,
                randomized_signals,
            )
            exceedances[horizon] += int(
                np.sum(np.isfinite(statistics) & (statistics >= analysis.granger_f))
            )
    return {
        horizon: (count + 1.0) / (config.permutations + 1.0)
        for horizon, count in exceedances.items()
    }


def _randomized_multivariate_statistics(
    analysis: _FactorAnalysis,
    generator: np.random.Generator,
    *,
    draws: int,
    batch_size: int,
    rhos: np.ndarray | None = None,
    confounder_errors: np.ndarray | None = None,
    confounder_scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    statistics = np.empty(draws, dtype=float)
    if rhos is not None and np.asarray(rhos).shape != (draws,):
        raise ValueError("rhos must contain one value per draw")
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        batch_rhos = None if rhos is None else np.asarray(rhos)[start:stop]
        randomized_signals = _simulate_conditional_signals(
            analysis.model,
            analysis.outcome,
            analysis.signal,
            generator,
            draws=stop - start,
            rhos=batch_rhos,
            confounder_errors=confounder_errors,
            confounder_scale=confounder_scale,
        )
        statistics[start:stop] = _batch_multivariate_granger_f_statistics(
            analysis.granger_design,
            randomized_signals,
        )
    return statistics


def _outcome_confounder_residuals(
    outcome: np.ndarray,
    *,
    lags: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the horizon-offset outcome AutoReg used by the notebook sensitivity step."""
    time_count, asset_count = outcome.shape
    offsets = np.arange(horizon, horizon + lags)
    first_origin = int(offsets.max())
    origins = np.arange(first_origin, time_count)
    fixed_effects = _fixed_effect_design(asset_count, len(origins))
    predictors = np.column_stack([outcome[origins - lag].reshape(-1) for lag in offsets])
    design = np.column_stack([fixed_effects, predictors])
    response = outcome[origins].reshape(-1)
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ coefficients
    residual_panel = np.zeros_like(outcome)
    residual_panel[origins] = residual.reshape(len(origins), asset_count)
    scale = np.sqrt(np.mean(residual_panel[origins] ** 2, axis=0))
    return residual_panel, scale


def _sensitivity_rho_star(
    analysis: _FactorAnalysis,
    config: ScreeningConfig,
    generator: np.random.Generator,
) -> float:
    """Notebook-style kernel sensitivity curve over treatment/outcome error correlation."""
    if config.sensitivity_draws == 0:
        return 1.0
    confounder_errors, confounder_scale = _outcome_confounder_residuals(
        analysis.outcome,
        lags=config.var_lags,
        horizon=config.forward_horizon,
    )
    rhos = np.linspace(-0.99, 0.99, config.sensitivity_draws)
    statistics = _randomized_multivariate_statistics(
        analysis,
        generator,
        draws=config.sensitivity_draws,
        batch_size=config.simulation_batch_size,
        rhos=rhos,
        confounder_errors=confounder_errors,
        confounder_scale=confounder_scale,
    )
    exceedances = (np.isfinite(statistics) & (statistics >= analysis.granger_f)).astype(float)

    # The notebook fits a local-constant KernelReg to these Bernoulli exceedances.
    # This explicit Gaussian smoother is deterministic and avoids bandwidth optimizer
    # failures on short, nearly separated sensitivity curves.
    spacing = 1.98 / max(config.sensitivity_draws - 1, 1)
    bandwidth = max(3.0 * spacing, 0.08)
    distance = (rhos[:, None] - rhos[None, :]) / bandwidth
    weights = np.exp(-0.5 * distance**2)
    fitted_p = weights @ exceedances / weights.sum(axis=1)
    overturned = np.flatnonzero(fitted_p > config.granger_p_threshold)
    if len(overturned) == 0:
        return 1.0
    return float(np.min(np.abs(rhos[overturned])))


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _multivariate_effect_diagnostics(model: _MultivariateVAR) -> dict[str, float | int | str]:
    direct_norm = float(np.linalg.norm(model.signal_own_lags[:, 0]))
    cross_maps = []
    outcome_loadings = model.outcome_projection.loadings
    signal_loadings = model.signal_projection.loadings
    inverse_signal_scale = np.diag(1.0 / model.signal_projection.scale)
    for coefficients in model.signal_to_outcome_lags:
        effect_map = outcome_loadings @ coefficients @ signal_loadings.T @ inverse_signal_scale
        effect_map = effect_map.copy()
        np.fill_diagonal(effect_map, 0.0)
        cross_maps.append(effect_map)
    cross_norm = float(np.sqrt(sum(np.linalg.norm(effect_map) ** 2 for effect_map in cross_maps)))

    conditional_beta = model.conditional_beta.copy()
    total_conditioning_norm = float(np.linalg.norm(conditional_beta))
    np.fill_diagonal(conditional_beta, 0.0)
    cross_conditioning_share = (
        float(np.linalg.norm(conditional_beta) / total_conditioning_norm)
        if total_conditioning_norm > 1e-14
        else 0.0
    )
    return {
        "causal_model": "reduced_rank_multivariate_var",
        "common_factor_count": model.outcome_projection.factor_count,
        "direct_signal_effect_norm": direct_norm,
        "cross_asset_signal_effect_norm": cross_norm,
        "assignment_cross_asset_share": cross_conditioning_share,
    }


def screen_factor_horizons(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    config: ScreeningConfig,
    *,
    seed_label: str = "screen",
) -> FactorScreeningResult:
    """Screen each factor over the configured effect horizons."""
    records: list[dict[str, object]] = []
    analyses: dict[tuple[str, int], _FactorAnalysis] = {}
    outcome_panel = _stationarize_panel(
        np.log(close.reindex(train_dates)),
        config,
        label="log-price",
        force_difference=True,
    )

    for factor_name, full_values in factors.items():
        factor_train = full_values.reindex(train_dates)
        try:
            base_analysis = _analyze_factor(
                factor_train,
                close,
                train_dates,
                config,
                outcome_panel,
            )
            factor_analyses = {
                horizon: _analysis_for_horizon(base_analysis, horizon, config)
                for horizon in config.effect_horizons
            }
            fisher_p_values = _fisher_horizon_p_values(
                factor_analyses,
                config,
                np.random.default_rng(
                    _stable_seed(config.seed, f"{seed_label}:{factor_name}:fisher")
                ),
            )
            analyses.update(
                {(factor_name, horizon): analysis for horizon, analysis in factor_analyses.items()}
            )
            model_diagnostics = _multivariate_effect_diagnostics(base_analysis.model)
            analysis_error = ""
        except (ValueError, np.linalg.LinAlgError) as error:
            factor_analyses = {}
            fisher_p_values = dict.fromkeys(config.effect_horizons, 1.0)
            analysis_error = str(error)
            model_diagnostics = {
                "causal_model": "reduced_rank_multivariate_var",
                "common_factor_count": 0,
                "direct_signal_effect_norm": np.nan,
                "cross_asset_signal_effect_norm": np.nan,
                "assignment_cross_asset_share": np.nan,
            }

        for horizon in config.effect_horizons:
            future = forward_returns(close, horizon).reindex(train_dates)
            rank_ic = daily_rank_ic(
                factor_train.shift(1),
                future,
                min_assets=config.min_assets,
            )
            mean_ic, ic_t, ic_p = newey_west_mean_test(rank_ic, max_lags=horizon)
            sign = float(np.sign(mean_ic)) if np.isfinite(mean_ic) else 0.0
            sign_consistency = (
                float((np.sign(rank_ic) == sign).mean()) if sign and len(rank_ic) else np.nan
            )
            analysis = factor_analyses.get(horizon)
            records.append(
                {
                    "factor": factor_name,
                    "horizon": horizon,
                    "mean_rank_ic": mean_ic,
                    "rank_ic_hac_t": ic_t,
                    "rank_ic_hac_p_value": ic_p,
                    "sign_consistency": sign_consistency,
                    "granger_f_statistic": analysis.granger_f if analysis else np.nan,
                    "granger_p_value": analysis.granger_p if analysis else 1.0,
                    "fisher_p_value": fisher_p_values[horizon],
                    "assignment_r_squared": (
                        analysis.model.signal_r_squared if analysis else np.nan
                    ),
                    "granger_restrictions": (
                        analysis.granger_design.degrees_numerator if analysis else 0
                    ),
                    "training_dates": len(rank_ic),
                    "panel_observations": analysis.panel_observations if analysis else 0,
                    "analysis_error": analysis_error,
                    "outcome_difference_order": (
                        analysis.outcome_difference_order if analysis else np.nan
                    ),
                    "signal_difference_order": (
                        analysis.signal_difference_order if analysis else np.nan
                    ),
                    "outcome_level_stationary_fraction": (
                        analysis.outcome_level_stationary_fraction if analysis else np.nan
                    ),
                    "signal_level_stationary_fraction": (
                        analysis.signal_level_stationary_fraction if analysis else np.nan
                    ),
                    "outcome_final_stationary_fraction": (
                        analysis.outcome_final_stationary_fraction if analysis else np.nan
                    ),
                    "signal_final_stationary_fraction": (
                        analysis.signal_final_stationary_fraction if analysis else np.nan
                    ),
                    **model_diagnostics,
                }
            )

    horizon_metrics = pd.DataFrame(records).set_index(["factor", "horizon"])
    horizon_metrics["fisher_q_value"] = 1.0
    for horizon in config.effect_horizons:
        horizon_slice = horizon_metrics.xs(horizon, level="horizon")["fisher_p_value"]
        adjusted = benjamini_hochberg(horizon_slice)
        for factor_name, q_value in adjusted.items():
            horizon_metrics.loc[(factor_name, horizon), "fisher_q_value"] = q_value

    # FDR (fisher_q_value) is reported as a diagnostic but is NOT a selection
    # gate; the sensitivity analysis (rho_star) is the focus. Selection requires
    # Granger p, raw Fisher p, rank IC, sign consistency, and min dates.
    initial_eligible = (
        (horizon_metrics["granger_p_value"] <= config.granger_p_threshold)
        & (horizon_metrics["fisher_p_value"] <= config.granger_p_threshold)
        & (horizon_metrics["mean_rank_ic"].abs() >= config.min_abs_rank_ic)
        & (horizon_metrics["sign_consistency"] >= config.min_sign_consistency)
        & (horizon_metrics["training_dates"] >= config.min_dates)
    )
    horizon_metrics["fisher_effective"] = initial_eligible
    horizon_metrics["confounding_rho_star"] = np.nan
    for factor_name, horizon in horizon_metrics.index[initial_eligible]:
        horizon_metrics.loc[(factor_name, horizon), "confounding_rho_star"] = (
            _sensitivity_rho_star(
                analyses[(factor_name, horizon)],
                config,
                np.random.default_rng(
                    _stable_seed(
                        config.seed,
                        f"{seed_label}:{factor_name}:horizon-{horizon}:sensitivity",
                    )
                ),
            )
        )
    horizon_metrics["sensitivity_rho_star"] = horizon_metrics["confounding_rho_star"]
    horizon_metrics["robust_effective"] = initial_eligible & (
        horizon_metrics["confounding_rho_star"].fillna(0.0) >= config.min_sensitivity_rho
    )

    primary = horizon_metrics.xs(config.forward_horizon, level="horizon").copy()
    duration_records = []
    for factor_name in primary.index:
        factor_horizons = horizon_metrics.xs(factor_name, level="factor")
        fisher_duration = 0
        robust_duration = 0
        for horizon in config.effect_horizons:
            if bool(factor_horizons.loc[horizon, "fisher_effective"]):
                fisher_duration = horizon
            else:
                break
        for horizon in config.effect_horizons:
            if bool(factor_horizons.loc[horizon, "robust_effective"]):
                robust_duration = horizon
            else:
                break
        robust_horizons = factor_horizons.index[factor_horizons["robust_effective"]]
        duration_records.append(
            {
                "factor": factor_name,
                "fisher_effect_horizon_sessions": fisher_duration,
                "robust_effect_horizon_sessions": robust_duration,
                "first_robust_horizon": (
                    int(robust_horizons.min()) if len(robust_horizons) else 0
                ),
                "last_robust_horizon": (
                    int(robust_horizons.max()) if len(robust_horizons) else 0
                ),
            }
        )
    durations = pd.DataFrame(duration_records).set_index("factor")
    metrics = primary.join(durations)
    metrics["selected"] = False
    eligible = metrics["robust_effective"]
    ranking = metrics.loc[eligible].assign(abs_rank_ic=metrics.loc[eligible, "mean_rank_ic"].abs())
    selected = (
        ranking.sort_values(
            ["confounding_rho_star", "abs_rank_ic"],
            ascending=[False, False],
        )
        .head(config.max_factors)
        .index
    )
    metrics.loc[selected, "selected"] = True
    return FactorScreeningResult(
        metrics=metrics.reset_index(),
        horizon_metrics=horizon_metrics.reset_index(),
    )


def screen_factors(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    config: ScreeningConfig,
    *,
    seed_label: str = "screen",
) -> pd.DataFrame:
    return screen_factor_horizons(
        factors,
        close,
        train_dates,
        config,
        seed_label=seed_label,
    ).metrics


def make_walk_forward_folds(
    index: pd.DatetimeIndex,
    *,
    warmup_sessions: int,
    train_sessions: int,
    test_sessions: int,
    forward_horizon: int,
) -> list[FoldDefinition]:
    first_test = warmup_sessions + train_sessions + forward_horizon
    folds: list[FoldDefinition] = []
    test_start = first_test
    fold_number = 1
    while test_start + test_sessions <= len(index):
        train_end = test_start - forward_horizon
        train_start = train_end - train_sessions
        folds.append(
            FoldDefinition(
                fold=fold_number,
                train_dates=index[train_start:train_end],
                embargo_dates=index[train_end:test_start],
                test_dates=index[test_start : test_start + test_sessions],
            )
        )
        fold_number += 1
        test_start += test_sessions
    return folds


def _selection_summary(metrics: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    summary = (
        metrics.groupby("factor")
        .agg(
            times_selected=("selected", "sum"),
            mean_rank_ic=("mean_rank_ic", "mean"),
            median_granger_p=("granger_p_value", "median"),
            median_fisher_q=("fisher_q_value", "median"),
            median_sensitivity_rho=("sensitivity_rho_star", "median"),
            median_fisher_effect_horizon=("fisher_effect_horizon_sessions", "median"),
            median_robust_effect_horizon=("robust_effect_horizon_sessions", "median"),
            mean_sign_consistency=("sign_consistency", "mean"),
        )
        .reset_index()
    )
    summary["selection_rate"] = summary["times_selected"] / fold_count
    return summary.sort_values(["times_selected", "median_fisher_q"], ascending=[False, True])


def walk_forward_causal_screen(
    frames: dict[str, pd.DataFrame],
    candidate_symbols: list[str],
    *,
    benchmark: str = "SPY",
    factor_names: list[str] | None = None,
    config: ScreeningConfig | None = None,
    initial_capital: float = 1_000_000.0,
    commission_bps: float = 5.0,
) -> WalkForwardResult:
    config = config or ScreeningConfig()
    benchmark = benchmark.upper()
    candidates = list(
        dict.fromkeys(symbol.upper() for symbol in candidate_symbols if symbol.upper() != benchmark)
    )
    if len(candidates) < config.min_assets:
        raise ValueError(
            f"At least {config.min_assets} non-benchmark assets are required; got {len(candidates)}"
        )
    if benchmark not in {symbol.upper() for symbol in frames}:
        raise ValueError(f"Benchmark {benchmark!r} is missing from market data")

    selected_names = factor_names or FACTOR_REGISTRY.names()
    all_factors = compute_factor_zoo(frames, selected_names, benchmark=benchmark)
    factors = {name: all_factors[name].reindex(columns=candidates) for name in selected_names}
    prices = close_matrix(frames)
    candidate_prices = prices.reindex(columns=candidates)
    warmup = max(FACTOR_REGISTRY.get(name).lookback for name in selected_names)
    folds = make_walk_forward_folds(
        prices.index,
        warmup_sessions=warmup,
        train_sessions=config.train_sessions,
        test_sessions=config.test_sessions,
        forward_horizon=config.forward_horizon,
    )
    if not folds:
        required = warmup + config.train_sessions + config.forward_horizon + config.test_sessions
        raise ValueError(
            f"Not enough history for one fold: need at least {required} aligned sessions, "
            f"got {len(prices.index)}"
        )

    metric_frames: list[pd.DataFrame] = []
    horizon_metric_frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    weight_frames: list[pd.DataFrame] = []

    for fold in folds:
        screening = screen_factor_horizons(
            factors,
            candidate_prices,
            fold.train_dates,
            config,
            seed_label=f"{fold.train_dates[0].date()}:{fold.train_dates[-1].date()}",
        )
        metrics = screening.metrics
        metrics.insert(0, "fold", fold.fold)
        metric_frames.append(metrics)
        horizon_metrics = screening.horizon_metrics
        horizon_metrics.insert(0, "fold", fold.fold)
        horizon_metric_frames.append(horizon_metrics)
        chosen = metrics.loc[metrics["selected"]].copy()

        target_dates = rebalance_dates(fold.test_dates, config.rebalance_frequency).union(
            pd.DatetimeIndex([fold.test_dates[0]])
        )
        if chosen.empty:
            fold_weights = pd.DataFrame(0.0, index=target_dates, columns=candidates)
        else:
            magnitudes = chosen["mean_rank_ic"].abs()
            factor_weights = magnitudes / magnitudes.sum()
            composite = pd.DataFrame(0.0, index=prices.index, columns=candidates)
            for (_, row), factor_weight in zip(chosen.iterrows(), factor_weights, strict=True):
                sign = np.sign(row["mean_rank_ic"])
                composite += cross_sectional_rank(factors[row["factor"]]) * sign * factor_weight
            fold_weights = rank_target_weights(
                composite,
                candidate_prices,
                top_n=min(config.top_n, len(candidates)),
                frequency=config.rebalance_frequency,
                signal_lag=1,
                long_short=config.long_short,
                target_dates=target_dates,
            )
        fold_weights = fold_weights.reindex(columns=prices.columns, fill_value=0.0)
        weight_frames.append(fold_weights)
        fold_records.append(
            {
                "fold": fold.fold,
                "train_start": fold.train_dates[0],
                "train_end": fold.train_dates[-1],
                "embargo_start": fold.embargo_dates[0],
                "embargo_end": fold.embargo_dates[-1],
                "test_start": fold.test_dates[0],
                "test_end": fold.test_dates[-1],
                "selected_count": len(chosen),
                "selected_factors": ",".join(chosen["factor"]),
            }
        )

    screening_metrics = pd.concat(metric_frames, ignore_index=True)
    horizon_metrics = pd.concat(horizon_metric_frames, ignore_index=True)
    targets = pd.concat(weight_frames).sort_index()
    targets = targets[~targets.index.duplicated(keep="last")]
    first_test = folds[0].test_dates[0]
    last_test = folds[-1].test_dates[-1]
    backtest_prices = prices.loc[first_test:last_test]
    backtest = run_weight_backtest(
        backtest_prices,
        targets,
        name="causal_factor_composite",
        benchmark_symbol=benchmark,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
    )

    fold_frame = pd.DataFrame(fold_records)
    equity = backtest.equity_curve
    for row_index, row in fold_frame.iterrows():
        start_position = equity.index.searchsorted(row["test_start"])
        base_position = max(start_position - 1, 0)
        end_position = equity.index.searchsorted(row["test_end"], side="right") - 1
        strategy_return = (
            equity.iloc[end_position]["causal_factor_composite"]
            / equity.iloc[base_position]["causal_factor_composite"]
            - 1.0
        )
        benchmark_name = f"buy_hold_{benchmark}"
        benchmark_return = (
            equity.iloc[end_position][benchmark_name] / equity.iloc[base_position][benchmark_name]
            - 1.0
        )
        fold_frame.loc[row_index, "strategy_return"] = strategy_return
        fold_frame.loc[row_index, "benchmark_return"] = benchmark_return
        fold_frame.loc[row_index, "excess_return"] = strategy_return - benchmark_return

    return WalkForwardResult(
        screening_metrics=screening_metrics,
        horizon_metrics=horizon_metrics,
        folds=fold_frame,
        selection_summary=_selection_summary(screening_metrics, len(folds)),
        target_weights=targets,
        backtest=backtest,
        run_config={
            **asdict(config),
            "screening_method": "analysis_multivariate_horizon_fisher_v2",
            "candidate_symbols": candidates,
            "benchmark": benchmark,
            "factor_names": selected_names,
            "data_start": str(prices.index.min().date()),
            "data_end": str(prices.index.max().date()),
            "initial_capital": initial_capital,
            "commission_bps": commission_bps,
        },
    )
