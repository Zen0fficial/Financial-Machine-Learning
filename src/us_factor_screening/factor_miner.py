"""Genetic programming factor mining via gplearn.

This module evolves new factor formulas from OHLCV + option-volume panels
using the Kakushadze operator vocabulary exposed in :mod:`alpha_operators`.

Design overview (Approach A - custom evaluation loop)
-----------------------------------------------------

gplearn's public ``SymbolicRegressor.fit`` validates ``X`` through scikit-learn's
``validate_data`` which rejects object-dtype arrays carrying DataFrames. We
therefore bypass the public estimator API and drive gplearn's ``_Program``
class directly:

1. The panel is represented as a 2D object numpy array ``X`` of shape
   ``(1, n_terminals)``. Each column holds a single ``Date x Symbol``
   ``DataFrame`` (the same reference is reused for the lone sample row).
2. Each Kakushadze operator is wrapped so it accepts 1D object arrays
   (the terminal / intermediate values produced by ``_Program.execute``),
   extracts the underlying ``DataFrame`` (or float constant), applies the
   operator, and returns the result wrapped in a fresh 1D object array.
3. A custom ``_Fitness`` metric evaluates the program output: it extracts
   the resulting ``DataFrame`` and computes the mean daily cross-sectional
   Spearman rank correlation with forward returns (``|mean RankIC|``).
4. A lightweight evolutionary loop (tournament selection + crossover /
   subtree / hoist / point mutation / reproduction) evolves the population.
5. Surviving programs are filtered by ``|RankIC|`` and cross-factor
   correlation to enforce diversity before being returned as
   :class:`MinedFactor` records.

Reference: Kakushadze, Z. (2016). 101 Formulaic Alphas. arXiv:1601.00991.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# gplearn internals (the public estimator API rejects object-dtype X).
from gplearn._program import _Program
from gplearn.fitness import _Fitness
from gplearn.functions import _Function
from scipy.stats import spearmanr

from .alpha_operators import (
    abs as gp_abs,
)
from .alpha_operators import (
    add as gp_add,
)
from .alpha_operators import (
    decay_linear,
    delay,
    delta,
    greater,
    less,
    max_,
    min_,
    rank,
    scale,
    sign,
    ts_arg_max,
    ts_arg_min,
    ts_corr,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std_dev,
    zscore,
)
from .alpha_operators import (
    divide as gp_divide,
)
from .alpha_operators import (
    log as gp_log,
)
from .alpha_operators import (
    multiply as gp_multiply,
)
from .alpha_operators import (
    subtract as gp_subtract,
)
from .factor_zoo import FACTOR_REGISTRY, compute_factor_zoo

__all__ = [
    "MiningConfig",
    "MinedFactor",
    "load_panel_data",
    "compute_existing_factors",
    "prepare_terminals",
    "make_fitness_function",
    "define_function_set",
    "mine_factors",
    "format_program",
    "evaluate_program",
    "save_mined_factors",
]


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclass
class MiningConfig:
    """Hyperparameters for the genetic programming factor mining loop."""

    population_size: int = 1000
    generations: int = 20
    tournament_size: int = 20
    max_depth: int = 8
    init_depth: tuple[int, int] = (2, 6)
    parsimony_coefficient: float = 0.001
    p_crossover: float = 0.7
    p_subtree_mutation: float = 0.1
    p_hoist_mutation: float = 0.05
    p_point_mutation: float = 0.1
    p_point_replace: float = 0.05
    n_factors: int = 10
    rankic_threshold: float = 0.02
    correlation_threshold: float = 0.7
    forward_return_horizon: int = 1
    random_state: int = 42


@dataclass
class MinedFactor:
    """A single mined factor with its formula and evaluation metrics."""

    formula: str
    rankic: float
    rankic_ir: float
    fitness: float
    program: object


# ---------------------------------------------------------------------------
# Panel data loading
# ---------------------------------------------------------------------------


def load_panel_data(
    ohlcv_path: str,
    option_volume_path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV + option volume into Date x Symbol DataFrames.

    Parameters
    ----------
    ohlcv_path
        Path to a long-format CSV with columns
        ``Date, Open, High, Low, Close, Volume, Symbol``.
    option_volume_path
        Optional path to a CSV (possibly ``.gz``) with at least
        ``Date, Symbol, OptionVolume`` columns.
    start_date, end_date
        Optional inclusive ISO-date bounds applied to the panel index.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps ``"open"``, ``"high"``, ``"low"``, ``"close"``, ``"volume"``,
        ``"returns"`` and (when available) ``"option_volume"`` to
        ``Date x Symbol`` DataFrames.
    """
    ohlcv = pd.read_csv(ohlcv_path)
    if "Date" not in ohlcv.columns:
        raise ValueError(f"{ohlcv_path}: expected a 'Date' column")

    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"], errors="coerce", utc=True)
    ohlcv["Date"] = ohlcv["Date"].dt.tz_localize(None)
    ohlcv = ohlcv.dropna(subset=["Date"]).sort_values("Date")

    if start_date is not None:
        ohlcv = ohlcv[ohlcv["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        ohlcv = ohlcv[ohlcv["Date"] <= pd.Timestamp(end_date)]

    panel: dict[str, pd.DataFrame] = {}
    for field_name in ("open", "high", "low", "close", "volume"):
        panel[field_name] = (
            ohlcv.pivot(index="Date", columns="Symbol", values=field_name.capitalize())
            .sort_index()
            .astype(float)
        )

    panel["returns"] = panel["close"].pct_change(fill_method=None)

    if option_volume_path:
        opt = pd.read_csv(option_volume_path)
        opt["Date"] = pd.to_datetime(opt["Date"], errors="coerce", utc=True)
        opt["Date"] = opt["Date"].dt.tz_localize(None)
        opt = opt.dropna(subset=["Date"])
        volume_col = "OptionVolume" if "OptionVolume" in opt.columns else opt.columns[2]
        opt_panel = (
            opt.pivot(index="Date", columns="Symbol", values=volume_col)
            .sort_index()
            .astype(float)
        )
        panel["option_volume"] = opt_panel.reindex(
            index=panel["close"].index, columns=panel["close"].columns
        )

    return panel


# ---------------------------------------------------------------------------
# Existing factor computation
# ---------------------------------------------------------------------------


def _panel_to_frames(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Convert a Date x Symbol panel to the per-symbol frames used by ``compute_factor_zoo``."""
    close = panel["close"]
    frames: dict[str, pd.DataFrame] = {}
    for symbol in close.columns:
        frames[symbol] = pd.DataFrame(
            {
                "Date": close.index,
                "Open": panel["open"][symbol].to_numpy(),
                "High": panel["high"][symbol].to_numpy(),
                "Low": panel["low"][symbol].to_numpy(),
                "Close": close[symbol].to_numpy(),
                "Volume": panel["volume"][symbol].to_numpy(),
            }
        )
    return frames


def _synthetic_benchmark_frames(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a synthetic SPY benchmark frame from the cross-sectional mean.

    The free Nasdaq-100 bundle does not include SPY; we use the equal-weighted
    cross-section as a market proxy so benchmark-dependent factors can still
    be computed. This is a research convenience, not a tradable index.
    """
    close = panel["close"]
    return pd.DataFrame(
        {
            "Date": close.index,
            "Open": panel["open"].mean(axis=1).to_numpy(),
            "High": panel["high"].mean(axis=1).to_numpy(),
            "Low": panel["low"].mean(axis=1).to_numpy(),
            "Close": close.mean(axis=1).to_numpy(),
            "Volume": panel["volume"].sum(axis=1).to_numpy(),
        }
    )


def compute_existing_factors(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute all registered factor-zoo factors as additional terminals.

    Returns a dict mapping factor name to a ``Date x Symbol`` DataFrame.
    Benchmark-dependent factors use a synthetic equal-weighted SPY proxy when
    SPY is not present in the panel, and option-volume factors are included
    only when ``panel["option_volume"]`` is available.
    """
    frames = _panel_to_frames(panel)

    if "SPY" not in panel["close"].columns:
        frames["SPY"] = _synthetic_benchmark_frames(panel)

    option_volume_frames: dict[str, pd.DataFrame] | None = None
    if "option_volume" in panel:
        opt = panel["option_volume"]
        option_volume_frames = {}
        for symbol in opt.columns:
            option_volume_frames[symbol] = pd.DataFrame(
                {
                    "Date": opt.index,
                    "OptionVolume": opt[symbol].to_numpy(),
                }
            )

    names = FACTOR_REGISTRY.names()
    if option_volume_frames is None:
        names = [n for n in names if "option_volume" not in n]

    factors = compute_factor_zoo(
        frames,
        names,
        benchmark="SPY",
        option_volume_frames=option_volume_frames,
    )

    # Reindex to the original panel's close shape, dropping the synthetic SPY
    # benchmark column so factors align with the raw OHLCV terminals.
    template = panel["close"]
    return {
        name: values.reindex(index=template.index, columns=template.columns)
        for name, values in factors.items()
    }


def prepare_terminals(
    panel: dict[str, pd.DataFrame],
    existing_factors: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Combine raw OHLCV + existing factors into the terminal set.

    Each terminal is a ``Date x Symbol`` DataFrame aligned to the panel close.
    """
    terminals: dict[str, pd.DataFrame] = {}
    for field_name in ("open", "high", "low", "close", "volume", "returns"):
        if field_name in panel:
            terminals[field_name] = panel[field_name]
    if "option_volume" in panel:
        terminals["option_volume"] = panel["option_volume"]

    if existing_factors:
        template = panel["close"]
        for name, values in existing_factors.items():
            aligned = values.reindex(index=template.index, columns=template.columns)
            terminals[name] = aligned

    return terminals


# ---------------------------------------------------------------------------
# gplearn function wrappers
# ---------------------------------------------------------------------------


def _extract_value(arg: object) -> object:
    """Extract a DataFrame or float from a gplearn terminal / intermediate value."""
    if isinstance(arg, np.ndarray):
        if arg.size == 0:
            return np.nan
        return arg.flat[0]
    return arg


def _wrap_result(result: object) -> np.ndarray:
    """Wrap a DataFrame or scalar into a 1D object array for gplearn."""
    out = np.empty(1, dtype=object)
    out[0] = result
    return out


def _safe_unary(op: Callable[[object], object]) -> Callable[[object], np.ndarray]:
    """Wrap a unary operator so it accepts gplearn 1D arrays and never raises."""

    def wrapped(x: object) -> np.ndarray:
        val = _extract_value(x)
        try:
            result = op(val)
        except Exception:
            result = np.nan
        return _wrap_result(result)

    return wrapped


def _safe_binary(op: Callable[[object, object], object]) -> Callable[[object, object], np.ndarray]:
    """Wrap a binary operator so it accepts gplearn 1D arrays and never raises."""

    def wrapped(x1: object, x2: object) -> np.ndarray:
        a = _extract_value(x1)
        b = _extract_value(x2)
        try:
            result = op(a, b)
        except Exception:
            result = np.nan
        return _wrap_result(result)

    return wrapped


def _unary_with_window(op: Callable[[pd.DataFrame, int], pd.DataFrame], n: int) -> Callable[[object], pd.DataFrame]:
    def fn(df: object) -> object:
        if not isinstance(df, pd.DataFrame):
            return np.nan
        return op(df, n)

    return fn


def _binary_with_window(
    op: Callable[[pd.DataFrame, pd.DataFrame, int], pd.DataFrame], n: int
) -> Callable[[object, object], object]:
    def fn(df1: object, df2: object) -> object:
        if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
            return np.nan
        return op(df1, df2, n)

    return fn


def _make_gplearn_function(name: str, arity: int, fn: Callable[..., object]) -> _Function:
    """Create a gplearn ``_Function`` bypassing ``make_function`` validation.

    ``make_function`` validates the callable against numeric ``np.ones(10)``
    arrays and rejects functions that operate on DataFrames. We construct the
    ``_Function`` directly so our DataFrame-aware wrappers are accepted.
    """
    return _Function(function=fn, name=name, arity=arity)


# Fixed window sizes for parameterised operators.
_TS_WINDOWS_SHORT = (1, 5, 10, 21)
_TS_WINDOWS_MEDIUM = (5, 10, 20, 60)
_TS_WINDOWS_LONG = (5, 20, 60)
_TS_WINDOWS_RANK = (5, 10, 20)
_TS_WINDOWS_CORR = (5, 10, 20)


def define_function_set() -> list[_Function]:
    """Return the gplearn function set built from the Kakushadze operators.

    Parameterised operators (e.g. ``ts_mean(df, n)``) are instantiated with
    several fixed window sizes, producing one gplearn function per variant
    (``ts_mean_5``, ``ts_mean_10``, ...). Binary operators like ``ts_corr``
    take two DataFrame arguments and are exposed with fixed windows too.
    """
    functions: list[_Function] = []

    # Unary cross-sectional / element-wise
    functions.append(_make_gplearn_function("rank", 1, _safe_unary(rank)))
    functions.append(_make_gplearn_function("scale", 1, _safe_unary(scale)))
    functions.append(_make_gplearn_function("zscore", 1, _safe_unary(zscore)))
    functions.append(_make_gplearn_function("sign", 1, _safe_unary(sign)))
    functions.append(_make_gplearn_function("abs", 1, _safe_unary(gp_abs)))
    functions.append(_make_gplearn_function("log", 1, _safe_unary(gp_log)))

    # Binary arithmetic
    functions.append(_make_gplearn_function("add", 2, _safe_binary(gp_add)))
    functions.append(_make_gplearn_function("subtract", 2, _safe_binary(gp_subtract)))
    functions.append(_make_gplearn_function("multiply", 2, _safe_binary(gp_multiply)))
    functions.append(_make_gplearn_function("divide", 2, _safe_binary(gp_divide)))
    functions.append(_make_gplearn_function("max_", 2, _safe_binary(max_)))
    functions.append(_make_gplearn_function("min_", 2, _safe_binary(min_)))
    functions.append(_make_gplearn_function("less", 2, _safe_binary(less)))
    functions.append(_make_gplearn_function("greater", 2, _safe_binary(greater)))

    # Time-series operators with fixed windows
    for n in _TS_WINDOWS_SHORT:
        functions.append(_make_gplearn_function(f"delay_{n}", 1, _safe_unary(_unary_with_window(delay, n))))
        functions.append(_make_gplearn_function(f"delta_{n}", 1, _safe_unary(_unary_with_window(delta, n))))

    for n in _TS_WINDOWS_MEDIUM:
        functions.append(_make_gplearn_function(f"ts_mean_{n}", 1, _safe_unary(_unary_with_window(ts_mean, n))))
        functions.append(_make_gplearn_function(f"ts_std_dev_{n}", 1, _safe_unary(_unary_with_window(ts_std_dev, n))))

    for n in _TS_WINDOWS_LONG:
        functions.append(_make_gplearn_function(f"ts_max_{n}", 1, _safe_unary(_unary_with_window(ts_max, n))))
        functions.append(_make_gplearn_function(f"ts_min_{n}", 1, _safe_unary(_unary_with_window(ts_min, n))))

    for n in _TS_WINDOWS_RANK:
        functions.append(_make_gplearn_function(f"ts_rank_{n}", 1, _safe_unary(_unary_with_window(ts_rank, n))))
        functions.append(_make_gplearn_function(f"ts_arg_max_{n}", 1, _safe_unary(_unary_with_window(ts_arg_max, n))))
        functions.append(_make_gplearn_function(f"ts_arg_min_{n}", 1, _safe_unary(_unary_with_window(ts_arg_min, n))))
        functions.append(_make_gplearn_function(f"decay_linear_{n}", 1, _safe_unary(_unary_with_window(decay_linear, n))))

    for n in _TS_WINDOWS_CORR:
        functions.append(
            _make_gplearn_function(f"ts_corr_{n}", 2, _safe_binary(_binary_with_window(ts_corr, n)))
        )

    return functions


# ---------------------------------------------------------------------------
# Fitness: cross-sectional RankIC
# ---------------------------------------------------------------------------


def _compute_rankic(
    factor: pd.DataFrame, forward_returns: pd.DataFrame, min_dates: int = 10, min_symbols: int = 5
) -> tuple[float, float]:
    """Return ``(mean_rankic, rankic_ir)`` for a factor panel vs forward returns."""
    common_dates = factor.index.intersection(forward_returns.index)
    if len(common_dates) < min_dates:
        return 0.0, 0.0

    daily_ics: list[float] = []
    for date in common_dates:
        f_row = factor.loc[date]
        r_row = forward_returns.loc[date]
        common = f_row.index.intersection(r_row.index)
        if len(common) < min_symbols:
            continue
        f_vals = f_row.loc[common].to_numpy(dtype=float)
        r_vals = r_row.loc[common].to_numpy(dtype=float)
        valid = np.isfinite(f_vals) & np.isfinite(r_vals)
        if int(valid.sum()) < min_symbols:
            continue
        f_valid = f_vals[valid]
        r_valid = r_vals[valid]
        if f_valid.std() < 1e-12 or r_valid.std() < 1e-12:
            continue
        with np.errstate(invalid="ignore"):
            ic, _ = spearmanr(f_valid, r_valid)
        if np.isfinite(ic):
            daily_ics.append(float(ic))

    if len(daily_ics) < min_dates:
        return 0.0, 0.0

    mean_ic = float(np.mean(daily_ics))
    std_ic = float(np.std(daily_ics, ddof=1))
    ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
    return mean_ic, ir


def make_fitness_function(forward_returns: pd.DataFrame) -> _Fitness:
    """Create a gplearn ``_Fitness`` based on absolute mean cross-sectional RankIC."""

    fwd = forward_returns

    def _rankic_fitness(y: object, y_pred: object, sample_weight: object) -> float:
        if not isinstance(y_pred, np.ndarray) or y_pred.dtype != object or y_pred.size == 0:
            return 0.0
        factor = y_pred.flat[0]
        if not isinstance(factor, pd.DataFrame):
            return 0.0
        try:
            mean_ic, _ = _compute_rankic(factor, fwd)
        except Exception:
            return 0.0
        return abs(mean_ic)

    return _Fitness(function=_rankic_fitness, greater_is_better=True)


# ---------------------------------------------------------------------------
# Program evaluation and formatting
# ---------------------------------------------------------------------------


def evaluate_program(program: _Program, X: np.ndarray) -> pd.DataFrame | None:
    """Execute a gplearn program on the terminal array and return the factor DataFrame."""
    try:
        y_pred = program.execute(X)
    except Exception:
        return None
    if not isinstance(y_pred, np.ndarray) or y_pred.dtype != object or y_pred.size == 0:
        return None
    factor = y_pred.flat[0]
    return factor if isinstance(factor, pd.DataFrame) else None


def format_program(program: _Program) -> str:
    """Convert a gplearn program to a human-readable formula string.

    gplearn's ``_Program.__str__`` renders the tree in Lisp style using the
    function ``name`` attribute and ``X{i}`` / ``feature_names[i]`` for
    terminals. We reuse it but post-process the terminal labels to use the
    actual feature names when available.
    """
    return str(program)


def _factor_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Pearson correlation between two factor panels (NaN-safe)."""
    common_dates = a.index.intersection(b.index)
    common_symbols = a.columns.intersection(b.columns)
    if len(common_dates) < 10 or len(common_symbols) < 5:
        return 0.0
    a_vals = a.loc[common_dates, common_symbols].to_numpy(dtype=float).ravel()
    b_vals = b.loc[common_dates, common_symbols].to_numpy(dtype=float).ravel()
    valid = np.isfinite(a_vals) & np.isfinite(b_vals)
    if int(valid.sum()) < 30:
        return 0.0
    corr = float(np.corrcoef(a_vals[valid], b_vals[valid])[0, 1])
    return corr if np.isfinite(corr) else 0.0


# ---------------------------------------------------------------------------
# Evolutionary loop
# ---------------------------------------------------------------------------


def _build_program(
    function_set: list[_Function],
    arities: dict[int, list[_Function]],
    config: MiningConfig,
    n_features: int,
    random_state: np.random.RandomState,
    metric: _Fitness,
    feature_names: list[str],
    program: list | None = None,
) -> _Program:
    return _Program(
        function_set=function_set,
        arities=arities,
        init_depth=config.init_depth,
        init_method="half and half",
        n_features=n_features,
        const_range=None,
        metric=metric,
        p_point_replace=config.p_point_replace,
        parsimony_coefficient=config.parsimony_coefficient,
        random_state=random_state,
        feature_names=feature_names,
        program=program,
    )


def _evaluate_fitness(
    program: _Program,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> None:
    """Compute and store ``raw_fitness_`` and ``fitness_`` on the program."""
    try:
        program.raw_fitness_ = program.raw_fitness(X, y, sample_weight)
    except Exception:
        program.raw_fitness_ = 0.0
    program.fitness_ = program.fitness()


def _tournament_select(
    population: Sequence[_Program],
    tournament_size: int,
    random_state: np.random.RandomState,
) -> _Program:
    n = len(population)
    contenders = random_state.randint(0, n, size=tournament_size)
    fitness = [population[i].fitness_ for i in contenders]
    winner_idx = int(contenders[np.argmax(fitness)])
    return population[winner_idx]


def _evolve_population(
    population: list[_Program],
    config: MiningConfig,
    random_state: np.random.RandomState,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    function_set: list[_Function],
    arities: dict[int, list[_Function]],
    n_features: int,
    metric: _Fitness,
    feature_names: list[str],
) -> list[_Program]:
    method_probs = np.cumsum(
        [
            config.p_crossover,
            config.p_subtree_mutation,
            config.p_hoist_mutation,
            config.p_point_mutation,
        ]
    )

    new_population: list[_Program] = []
    for _ in range(config.population_size):
        parent = _tournament_select(population, config.tournament_size, random_state)
        method = random_state.uniform()

        try:
            if method < method_probs[0]:
                donor = _tournament_select(population, config.tournament_size, random_state)
                program_list, _, _ = parent.crossover(donor.program, random_state)
            elif method < method_probs[1]:
                program_list, _ = parent.subtree_mutation(random_state)
            elif method < method_probs[2]:
                program_list, _ = parent.hoist_mutation(random_state)
            elif method < method_probs[3]:
                program_list, _ = parent.point_mutation(random_state)
            else:
                program_list = parent.reproduce()
        except Exception:
            program_list = parent.reproduce()

        offspring = _build_program(
            function_set,
            arities,
            config,
            n_features,
            random_state,
            metric,
            feature_names,
            program=program_list,
        )
        _evaluate_fitness(offspring, X, y, sample_weight)
        new_population.append(offspring)

    return new_population


def mine_factors(
    panel: dict[str, pd.DataFrame],
    config: MiningConfig | None = None,
    existing_factors: dict[str, pd.DataFrame] | None = None,
) -> list[MinedFactor]:
    """Run the genetic programming factor mining loop.

    Parameters
    ----------
    panel
        Output of :func:`load_panel_data` - a dict of ``Date x Symbol``
        DataFrames with at least ``"close"`` and ``"returns"``.
    config
        Mining hyperparameters. A tiny default config
        (``population_size=50, generations=3``) is used when ``None`` so the
        function remains cheap to call from tests; supply a real
        :class:`MiningConfig` for production runs.
    existing_factors
        Pre-computed factor panels (from :func:`compute_existing_factors`)
        to include as terminals and to use for diversity filtering.

    Returns
    -------
    list[MinedFactor]
        The best diverse mined factors, sorted by descending ``|RankIC|``.
    """
    if config is None:
        config = MiningConfig(population_size=50, generations=3, n_factors=5)

    terminals = prepare_terminals(panel, existing_factors)
    terminal_names = list(terminals.keys())
    n_features = len(terminal_names)
    if n_features == 0:
        raise ValueError("No terminals available for mining")

    X = np.empty((1, n_features), dtype=object)
    for i, name in enumerate(terminal_names):
        X[0, i] = terminals[name]

    h = config.forward_return_horizon
    forward_returns = panel["close"].shift(-h) / panel["close"] - 1.0

    function_set = define_function_set()
    arities: dict[int, list[_Function]] = {}
    for fn in function_set:
        arities.setdefault(fn.arity, []).append(fn)

    metric = make_fitness_function(forward_returns)

    y = np.array([0.0])
    sample_weight = np.array([1.0])
    random_state = np.random.RandomState(config.random_state)

    # Initial population
    population: list[_Program] = []
    for _ in range(config.population_size):
        program = _build_program(
            function_set, arities, config, n_features, random_state, metric, terminal_names
        )
        _evaluate_fitness(program, X, y, sample_weight)
        population.append(program)

    # Evolution
    for _ in range(config.generations):
        population = _evolve_population(
            population,
            config,
            random_state,
            X,
            y,
            sample_weight,
            function_set,
            arities,
            n_features,
            metric,
            terminal_names,
        )

    population.sort(key=lambda p: p.fitness_, reverse=True)

    # Diversity filtering
    mined: list[MinedFactor] = []
    kept_factors: list[pd.DataFrame] = []
    if existing_factors:
        kept_factors.extend(existing_factors.values())

    for program in population:
        if len(mined) >= config.n_factors:
            break
        factor = evaluate_program(program, X)
        if factor is None:
            continue
        rankic, rankic_ir = _compute_rankic(factor, forward_returns)
        if abs(rankic) < config.rankic_threshold:
            continue
        if any(abs(_factor_correlation(factor, kept)) > config.correlation_threshold for kept in kept_factors):
            continue
        mined.append(
            MinedFactor(
                formula=format_program(program),
                rankic=rankic,
                rankic_ir=rankic_ir,
                fitness=float(program.fitness_),
                program=program,
            )
        )
        kept_factors.append(factor)

    mined.sort(key=lambda mf: abs(mf.rankic), reverse=True)
    return mined


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_mined_factors(
    mined: list[MinedFactor], output_dir: str | Path, config: MiningConfig | None = None
) -> Path:
    """Save mined factors (CSV summary + pickled programs) to ``output_dir``.

    Returns the path to the summary CSV.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "formula": mf.formula,
                "rankic": mf.rankic,
                "rankic_ir": mf.rankic_ir,
                "fitness": mf.fitness,
            }
            for mf in mined
        ]
    )
    csv_path = out / "mined_factors.csv"
    summary.to_csv(csv_path, index=False)

    programs_path = out / "mined_programs.pkl"
    with open(programs_path, "wb") as fh:
        pickle.dump(
            {
                "config": config,
                "programs": [mf.program for mf in mined],
                "formulas": [mf.formula for mf in mined],
            },
            fh,
        )

    return csv_path
