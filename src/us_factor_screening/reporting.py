from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .backtest import BacktestArtifacts
from .data import MarketDataPanel, _write_csv_atomic

if TYPE_CHECKING:
    from .causal_screening import WalkForwardResult


def write_data_outputs(
    frames: Mapping[str, pd.DataFrame],
    quality: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    panel = pd.concat(
        [frame.assign(Symbol=symbol) for symbol, frame in frames.items()],
        ignore_index=True,
    )
    data_path = target / "ohlcv.csv"
    _write_csv_atomic(panel, data_path)
    quality.to_csv(target / "data_quality.csv", index=False)
    if isinstance(frames, MarketDataPanel):
        manifest = frames.manifest()
        manifest["snapshot"] = {
            "file": data_path.name,
            "sha256": _sha256(data_path),
        }
        (target / "market_data_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_backtest_outputs(artifacts: BacktestArtifacts, output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts.stats.to_csv(target / "summary.csv")
    artifacts.equity_curve.to_csv(target / "equity_curve.csv")
    artifacts.target_weights.to_csv(target / "target_weights.csv")
    artifacts.realized_weights.to_csv(target / "realized_weights.csv")


def write_factor_outputs(
    factors: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    target = Path(output_dir)
    factor_dir = target / "factors"
    factor_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(target / "factor_manifest.csv", index=False)

    coverage_records = []
    for name, values in factors.items():
        values.to_csv(factor_dir / f"{name}.csv.gz", compression="gzip", index_label="Date")
        valid_dates = values.notna().any(axis=1)
        first_valid = values.index[valid_dates][0] if valid_dates.any() else None
        coverage_records.append(
            {
                "name": name,
                "observations": int(values.notna().sum().sum()),
                "possible_observations": int(values.size),
                "coverage": float(values.notna().sum().sum() / values.size),
                "first_valid_date": first_valid,
            }
        )
    pd.DataFrame(coverage_records).to_csv(target / "factor_coverage.csv", index=False)


def write_walk_forward_outputs(
    result: WalkForwardResult,
    output_dir: str | Path,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result.screening_metrics.to_csv(target / "screening_metrics.csv", index=False)
    result.horizon_metrics.to_csv(target / "horizon_metrics.csv", index=False)
    result.folds.to_csv(target / "walk_forward_folds.csv", index=False)
    result.selection_summary.to_csv(target / "selection_summary.csv", index=False)
    result.target_weights.to_csv(target / "walk_forward_target_weights.csv")
    (target / "run_config.json").write_text(
        json.dumps(result.run_config, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    write_backtest_outputs(result.backtest, target / "backtest")
