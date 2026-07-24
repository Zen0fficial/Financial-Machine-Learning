"""Local adapter for the AlphaGen RL factor-mining framework (KDD 2023).

AlphaGen (https://github.com/RL-MLDM/alphagen) is a reinforcement-learning
framework that mines formulaic alphas as token sequences. Its reference
implementation is tightly coupled to Qlib for data loading and target
construction. This adapter lets AlphaGen run on the local pandas/numpy
``Date x Symbol`` OHLCV panels used by :mod:`us_factor_screening` **without
installing Qlib**.

The bypass is non-invasive: AlphaGen's ``StockData.__init__`` calls
``_init_qlib()`` only when the module-level ``_QLIB_INITIALIZED`` flag is
``False``. Setting that flag to ``True`` before construction, and passing a
``preloaded_data`` tensor, skips every Qlib code path. AlphaGen's source is
imported unchanged.

Architecture summary (see the report for details):

* ``AlphaCalculator`` (abstract) requires ``calc_single_IC_ret``,
  ``calc_single_rIC_ret``, ``calc_mutual_IC``, ``calc_pool_IC_ret``,
  ``calc_pool_rIC_ret`` and ``calc_pool_all_ret``.
  ``TensorAlphaCalculator`` implements all of them given an
  ``evaluate_alpha(expr) -> Tensor(days, stocks)`` method plus a ``target``
  tensor.
* ``StockData`` holds a ``(n_days_total, n_features, n_stocks)`` tensor plus
  ``max_backtrack_days`` / ``max_future_days`` padding. ``Feature.evaluate``
  indexes ``data[start:stop, int(feature), :]``.
* The RL environment (``AlphaEnv``) builds token sequences action by action;
  when the ``SEP`` token is emitted the candidate ``Expression`` is evaluated
  by an ``AlphaPool`` whose ``try_new_expr`` returns the reward.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Make AlphaGen importable without modifying it, then neutralise its Qlib hook.
# ---------------------------------------------------------------------------

_ALPHAGEN_ROOT = Path(__file__).resolve().parents[2].parent / "alphagen"
if str(_ALPHAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALPHAGEN_ROOT))

# Flip the module-level flag before anything instantiates ``StockData`` so the
# lazy ``import qlib`` inside ``initialize_qlib`` is never reached.
import alphagen_qlib.stock_data as _ag_stock_data_module  # noqa: E402

_ag_stock_data_module._QLIB_INITIALIZED = True

from alphagen.data.calculator import TensorAlphaCalculator  # noqa: E402
from alphagen.data.expression import (  # noqa: E402
    Expression,
    Operators,
)
from alphagen.models.linear_alpha_pool import MseAlphaPool  # noqa: E402
from alphagen.rl.env.wrapper import AlphaEnv  # noqa: E402
from alphagen.utils.pytorch_utils import normalize_by_day  # noqa: E402
from alphagen_qlib.stock_data import StockData  # noqa: E402

# Re-export the most useful AlphaGen symbols for downstream consumers.
__all__ = [
    "ALPHAGEN_OPERATORS",
    "OPERATOR_TO_LOCAL",
    "LocalAlphaCalculator",
    "build_stock_data",
    "load_local_panel",
    "build_environment",
    "run_alphagen",
]

# ---------------------------------------------------------------------------
# Operator vocabulary
# ---------------------------------------------------------------------------

#: AlphaGen's own ``Operator`` subclasses used by the RL environment's token
#: builder. These are the canonical classes AlphaGen's ``ExpressionBuilder``
#: knows how to assemble; we reuse them verbatim so the env's action space and
#: validity checks work unchanged.
ALPHAGEN_OPERATORS: list[type] = list(Operators)

#: Mapping from AlphaGen operator class name to the equivalent function in
#: :mod:`us_factor_screening.alpha_operators`. AlphaGen evaluates expressions
#: itself on the ``StockData`` tensor, so this map is for *translation* of a
#: mined AlphaGen expression into the local pandas vocabulary (e.g. so the
#: mined formula can be re-evaluated against ``FactorContext`` afterwards).
OPERATOR_TO_LOCAL: dict[str, str] = {
    "Abs": "abs",
    "Sign": "sign",
    "Log": "log",
    "CSRank": "rank",
    "Add": "add",
    "Sub": "subtract",
    "Mul": "multiply",
    "Div": "divide",
    "Pow": "power",
    "Greater": "greater",
    "Less": "less",
    "Ref": "delay",
    "Mean": "ts_mean",
    "Sum": "ts_sum",
    "Std": "ts_std_dev",
    "Var": "ts_var",
    "Skew": "ts_skew",
    "Kurt": "ts_kurt",
    "Max": "ts_max",
    "Min": "ts_min",
    "Med": "ts_quantile",
    "Mad": None,
    "Rank": "ts_rank",
    "Delta": "delta",
    "WMA": "decay_linear",
    "EMA": None,
    "Cov": "ts_cov",
    "Corr": "ts_corr",
}

#: FeatureType order in the StockData tensor's feature axis. VWAP is synthesised
#: from OHLC because AlphaGen's ``FeatureType`` enum reserves a slot for it and
#: the RL action space can emit ``$vwap`` tokens.
_FEATURE_KEYS = ("open", "close", "high", "low", "volume")

_DEFAULT_MAX_BACKTRACK_DAYS = 100
_DEFAULT_FORWARD_HORIZON = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_local_panel(
    ohlcv_csv: str | os.PathLike[str] | None = None,
    *,
    forward_horizon: int = _DEFAULT_FORWARD_HORIZON,
    max_stocks: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load the free Nasdaq-100 OHLCV bundle into panel DataFrames.

    Returns ``(data, forward_returns)`` where ``data`` maps feature name
    (``open``/``high``/``low``/``close``/``volume``) to a ``Date x Symbol``
    DataFrame and ``forward_returns`` is the ``forward_horizon``-day forward
    close-to-close return aligned on the same axes.
    """
    if ohlcv_csv is None:
        ohlcv_csv = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "free_nasdaq100_2024_2026"
            / "ohlcv.csv"
        )
    raw = pd.read_csv(ohlcv_csv)
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize()
    raw = raw.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    symbols = sorted(raw["Symbol"].unique())
    if max_stocks is not None:
        symbols = symbols[:max_stocks]
    raw = raw[raw["Symbol"].isin(symbols)]

    data: dict[str, pd.DataFrame] = {}
    for field in _FEATURE_KEYS:
        data[field] = raw.pivot(index="Date", columns="Symbol", values=field.capitalize())
        data[field] = data[field].sort_index()
    data["volume"] = data["volume"].astype(float)

    close = data["close"]
    forward_returns = (close.shift(-forward_horizon) / close - 1).iloc[:-forward_horizon]
    return data, forward_returns


def build_stock_data(
    data: dict[str, pd.DataFrame],
    *,
    max_backtrack_days: int = _DEFAULT_MAX_BACKTRACK_DAYS,
    forward_horizon: int = _DEFAULT_FORWARD_HORIZON,
    device: torch.device | str = "cpu",
) -> StockData:
    """Build an AlphaGen :class:`StockData` from local pandas panels.

    The resulting tensor has shape
    ``(n_days, n_features, n_stocks)`` with the feature axis laid out in
    :class:`FeatureType` enum order (``OPEN, CLOSE, HIGH, LOW, VOLUME, VWAP``).
    ``max_future_days`` is set to ``forward_horizon`` so the effective
    ``n_days`` window excludes dates whose forward return is unavailable.
    """
    _ensure_qlib_bypassed()

    feature_frames = [data[k] for k in _FEATURE_KEYS]
    template = feature_frames[0]
    feature_frames = [f.reindex(index=template.index, columns=template.columns) for f in feature_frames]

    # Synthesise VWAP = (high + low + close) / 3 to populate FeatureType.VWAP.
    vwap = (data["high"] + data["low"] + data["close"]) / 3
    vwap = vwap.reindex(index=template.index, columns=template.columns)
    feature_frames.append(vwap)

    # Stack into (n_days, n_features, n_stocks). Feature order must match the
    # FeatureType IntEnum values: OPEN=0, CLOSE=1, HIGH=2, LOW=3, VOLUME=4, VWAP=5.
    stacked = np.stack([f.to_numpy(dtype=np.float64) for f in feature_frames], axis=1)
    tensor = torch.from_numpy(stacked).to(torch.device(device)).to(torch.float32)

    dates = template.index
    stock_ids = template.columns
    return StockData(
        instrument=list(stock_ids.astype(str)),
        start_time=str(dates[0].date()),
        end_time=str(dates[-1].date()),
        max_backtrack_days=max_backtrack_days,
        max_future_days=forward_horizon,
        device=torch.device(device),
        preloaded_data=(tensor, dates, stock_ids),
    )


def _ensure_qlib_bypassed() -> None:
    """Re-assert the Qlib bypass flag (idempotent, safe across reloads)."""
    _ag_stock_data_module._QLIB_INITIALIZED = True


# ---------------------------------------------------------------------------
# LocalAlphaCalculator
# ---------------------------------------------------------------------------


class LocalAlphaCalculator(TensorAlphaCalculator):
    """Evaluate AlphaGen expressions on local pandas/numpy panels.

    Subclasses :class:`TensorAlphaCalculator` so all IC / rank-IC / pool /
    mutual-IC computations are inherited. We only have to supply
    ``evaluate_alpha`` (which delegates to AlphaGen's own expression evaluator
    running on a Qlib-free :class:`StockData`) and the ``target`` tensor.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        *,
        max_backtrack_days: int = _DEFAULT_MAX_BACKTRACK_DAYS,
        forward_horizon: int = _DEFAULT_FORWARD_HORIZON,
        device: torch.device | str = "cpu",
    ) -> None:
        self._stock_data = build_stock_data(
            data,
            max_backtrack_days=max_backtrack_days,
            forward_horizon=forward_horizon,
            device=device,
        )
        target_tensor = self._build_target(forward_returns, device)
        super().__init__(target_tensor)

    # -- TensorAlphaCalculator interface -----------------------------------

    def evaluate_alpha(self, expr: Expression) -> torch.Tensor:
        return normalize_by_day(expr.evaluate(self._stock_data))

    @property
    def n_days(self) -> int:
        return self._stock_data.n_days

    @property
    def stock_data(self) -> StockData:
        return self._stock_data

    # -- helpers -----------------------------------------------------------

    def _build_target(
        self,
        forward_returns: pd.DataFrame,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Align ``forward_returns`` to the StockData effective date window.

        The StockData effective window covers dates
        ``[max_backtrack_days : max_backtrack_days + n_days]`` of the full
        tensor, i.e. it drops ``max_backtrack_days`` rows from the start and
        ``max_future_days`` (= ``forward_horizon``) rows from the end. We
        reindex the forward-return panel onto that exact window.
        """
        full_dates = self._stock_data._dates
        stock_ids = self._stock_data.stock_ids
        b = self._stock_data.max_backtrack_days
        n = self._stock_data.n_days
        effective_dates = full_dates[b : b + n]

        aligned = forward_returns.reindex(index=effective_dates, columns=stock_ids)
        target = torch.from_numpy(
            aligned.to_numpy(dtype=np.float64).copy()
        ).to(torch.device(device)).to(torch.float32)
        return normalize_by_day(target)


# ---------------------------------------------------------------------------
# RL environment + training
# ---------------------------------------------------------------------------


def build_environment(
    calculator: LocalAlphaCalculator | None = None,
    *,
    data: dict[str, pd.DataFrame] | None = None,
    forward_returns: pd.DataFrame | None = None,
    pool_capacity: int = 10,
    device: torch.device | str = "cpu",
    print_expr: bool = False,
) -> tuple[AlphaEnv, MseAlphaPool, LocalAlphaCalculator]:
    """Create the AlphaGen RL environment wired to local data.

    Returns ``(env, pool, calculator)``. If ``calculator`` is not supplied,
    ``data`` and ``forward_returns`` must be provided so a calculator can be
    built from scratch.
    """
    if calculator is None:
        if data is None or forward_returns is None:
            raise ValueError("Either pass a calculator or (data, forward_returns).")
        calculator = LocalAlphaCalculator(data, forward_returns, device=device)

    pool = MseAlphaPool(
        capacity=pool_capacity,
        calculator=calculator,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=torch.device(device),
    )
    env = AlphaEnv(pool=pool, device=torch.device(device), print_expr=print_expr)
    return env, pool, calculator


def run_alphagen(
    *,
    steps: int = 1000,
    pool_capacity: int = 10,
    max_stocks: int | None = 20,
    seed: int = 0,
    device: torch.device | str = "cpu",
    verbose: int = 0,
    ohlcv_csv: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Smoke-test the full AlphaGen RL pipeline on local data.

    Loads the local panel, builds a :class:`LocalAlphaCalculator`, wires up the
    ``AlphaEnv`` + an ``MseAlphaPool``, trains a masked-PPO agent for a small
    number of steps, and returns the mined factor expressions with their
    single-alpha ICs and the pool's ensemble IC.
    """
    from alphagen.rl.policy import LSTMSharedNet
    from alphagen.utils import reseed_everything
    from sb3_contrib.ppo_mask import MaskablePPO

    reseed_everything(seed)
    torch_device = torch.device(device)

    data, forward_returns = load_local_panel(ohlcv_csv, max_stocks=max_stocks)
    calculator = LocalAlphaCalculator(
        data, forward_returns, device=torch_device, forward_horizon=_DEFAULT_FORWARD_HORIZON
    )

    env, pool, _ = build_environment(
        calculator,
        pool_capacity=pool_capacity,
        device=torch_device,
        print_expr=verbose > 0,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": LSTMSharedNet,
            "features_extractor_kwargs": {
                "n_layers": 1,
                "d_model": 32,
                "dropout": 0.0,
                "device": torch_device,
            },
        },
        gamma=1.0,
        ent_coef=0.01,
        batch_size=64,
        n_steps=256,
        device=torch_device,
        verbose=verbose,
    )
    model.learn(total_timesteps=steps, progress_bar=False)

    return _summarize_pool(pool)


def _summarize_pool(pool: MseAlphaPool) -> dict[str, Any]:
    exprs = [str(e) for e in pool.exprs[: pool.size]]
    weights = [float(w) for w in pool.weights[: pool.size]]
    ics = [float(ic) for ic in pool.single_ics[: pool.size]]
    return {
        "n_factors": pool.size,
        "expressions": exprs,
        "weights": weights,
        "single_ics": ics,
        "best_ensemble_ic": float(pool.best_ic_ret),
        "eval_cnt": pool.eval_cnt,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke entry point
    import json

    result = run_alphagen(steps=500, max_stocks=10, verbose=1)
    print(json.dumps(result, indent=2))
