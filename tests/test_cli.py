from __future__ import annotations

from argparse import Namespace
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from us_factor_screening.cli import _argument_symbols, build_option_features_command, build_parser
from us_factor_screening.options import OptionChain, OptionObservation


def test_market_data_cli_defaults_to_polygon() -> None:
    args = build_parser().parse_args(
        [
            "validate-data",
            "--symbols",
            "EWY",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-03",
        ]
    )

    assert args.provider == "polygon"
    assert args.interval == "1d"
    assert args.alpaca_adjustment == "all"


def test_causal_cli_defaults_to_full_horizon_sensitivity() -> None:
    args = build_parser().parse_args(["causal-screen"])

    assert args.permutations == 999
    assert args.sensitivity_draws == 999
    assert args.effect_horizons == [1, 2, 3, 4, 5, 6, 7]
    assert args.stationarity_required_fraction == 0.80


def test_bivariate_rho_cli_defaults_to_requested_experiment() -> None:
    args = build_parser().parse_args(["bivariate-rho"])

    assert args.factor == "momentum_63d"
    assert args.analysis_start == "2024-01-01"
    assert args.analysis_end == "2026-12-31"
    assert args.sensitivity_draws == 999
    assert args.effect_horizons == [1, 2, 3, 4, 5, 6, 7]
    assert args.min_observations == 252
    assert args.workers == 0


def test_free_data_cli_defaults_to_free_history_bundle() -> None:
    args = build_parser().parse_args(["acquire-free-data"])

    assert args.start == "2024-01-01"
    assert args.min_price_sessions == 252
    assert args.max_start_gap_days == 10
    assert args.option_cache_dir == ".cache/cboe_option_volume"


def test_symbols_file_uses_only_eligible_rows(tmp_path) -> None:
    symbols_file = tmp_path / "universe.csv"
    pd.DataFrame(
        {"symbol": ["AAPL", "NEW"], "eligible": [True, False]}
    ).to_csv(symbols_file, index=False)

    args = build_parser().parse_args(
        ["validate-data", "--symbols-file", str(symbols_file)]
    )

    assert _argument_symbols(args) == ["AAPL"]


def test_option_feature_command_writes_normalized_daily_output(tmp_path) -> None:
    as_of = datetime(2024, 1, 3, 20, 46, tzinfo=UTC)
    observations = []
    for suffix, right, delta, iv, volume in (
        ("C", "call", 0.25, 0.20, 10),
        ("P", "put", -0.25, 0.28, 20),
    ):
        observations.append(
            OptionObservation(
                as_of=as_of,
                underlying="AAPL",
                contract=f"AAPL240202{suffix}00100000",
                expiration=date(2024, 1, 3) + timedelta(days=30),
                strike=100,
                right=right,
                provider="fixture",
                bid=1.0,
                ask=1.1,
                volume=volume,
                open_interest=100,
                underlying_price=100,
                implied_volatility=iv,
                delta=delta,
            )
        )
    source = tmp_path / "options.csv"
    output = tmp_path / "features.csv"
    OptionChain(
        observations=tuple(observations),
        provider="fixture",
        retrieved_at=as_of,
    ).to_frame().to_csv(source, index=False)

    exit_code = build_option_features_command(
        Namespace(
            input=source,
            output=output,
            option_provider=None,
            near_dte=30,
            far_dte=90,
            maximum_dte_distance=21,
            volume_zscore_window=20,
        )
    )

    features = pd.read_csv(output)
    assert exit_code == 0
    assert features.loc[0, "underlying"] == "AAPL"
    assert features.loc[0, "put_call_volume_ratio"] == 2.0
