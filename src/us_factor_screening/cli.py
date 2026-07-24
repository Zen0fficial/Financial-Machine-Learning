from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtest import run_weight_backtest
from .bivariate_sensitivity import BivariateSensitivityConfig, estimate_bivariate_rho
from .causal_screening import ScreeningConfig, walk_forward_causal_screen
from .data import (
    MarketDataPanel,
    MarketDataProvenance,
    MarketDataSource,
    YahooFinanceSource,
    close_matrix,
    validate_ohlcv,
)
from .factor_zoo import FACTOR_REGISTRY, compute_factor_zoo, orient_factor
from .factors import rank_target_weights
from .free_dataset import (
    CboeOptionVolumeSource,
    aggregate_cboe_option_volume,
    fetch_nasdaq100_snapshot,
    select_price_history_eligible,
    write_free_dataset_bundle,
)
from .options import OptionFeatureConfig, build_daily_option_features, load_option_chain
from .providers import (
    AlpacaMarketDataSource,
    FrozenMarketDataSource,
    PolygonMarketDataSource,
)
from .reporting import (
    write_backtest_outputs,
    write_data_outputs,
    write_factor_outputs,
    write_walk_forward_outputs,
)

DEFAULT_VALIDATION_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "BAC",
    "XOM",
    "CVX",
    "JNJ",
    "UNH",
    "PG",
    "KO",
    "HD",
    "CAT",
]


def _symbols(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))


def _argument_symbols(args: argparse.Namespace) -> list[str]:
    symbols_file = getattr(args, "symbols_file", None)
    if symbols_file is None:
        return _symbols(args.symbols)
    table = pd.read_csv(symbols_file)
    if "symbol" not in table.columns:
        raise SystemExit(f"{symbols_file} must contain a 'symbol' column")
    if "eligible" in table.columns:
        eligibility = table["eligible"]
        if eligibility.dtype == bool:
            eligible = eligibility
        else:
            eligible = eligibility.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
        table = table.loc[eligible]
    return _symbols(table["symbol"].dropna().astype(str).tolist())


def _source(args: argparse.Namespace) -> MarketDataSource:
    if args.provider == "polygon":
        return PolygonMarketDataSource(
            cache_dir=args.cache_dir,
            api_key=getattr(args, "api_key", None),
            interval=getattr(args, "interval", "1d"),
        )
    if args.provider == "yahoo":
        return YahooFinanceSource(cache_dir=args.cache_dir)
    if args.provider == "alpaca":
        return AlpacaMarketDataSource(
            cache_dir=args.cache_dir,
            feed=args.alpaca_feed,
            adjustment=args.alpaca_adjustment,
        )
    if args.source is None:
        raise SystemExit("--source is required when --provider=frozen")
    supplied = (args.source_provider, args.source_feed, args.source_adjustment)
    if any(supplied) and not all(supplied):
        raise SystemExit(
            "Frozen data without a complete manifest requires --source-provider, "
            "--source-feed, and --source-adjustment together"
        )
    provenance = None
    if all(supplied):
        provenance = MarketDataProvenance(
            provider=args.source_provider,
            feed=args.source_feed,
            adjustment=args.source_adjustment,
            interval="1d",
            session=args.source_session,
        )
    return FrozenMarketDataSource(args.source, provenance=provenance)


def _fetch(
    args: argparse.Namespace,
    *,
    include_benchmark: bool = False,
) -> MarketDataPanel:
    symbols = _argument_symbols(args)
    if include_benchmark:
        benchmark = args.benchmark.strip().upper()
        if benchmark not in symbols:
            symbols.append(benchmark)
    return _source(args).fetch_many(
        symbols,
        args.start,
        args.end,
        refresh=args.refresh,
    )


def validate_data_command(args: argparse.Namespace) -> int:
    frames = _fetch(args)
    quality = pd.DataFrame(
        [validate_ohlcv(frame, symbol).as_record() for symbol, frame in frames.items()]
    )
    write_data_outputs(frames, quality, args.output_dir)
    print(quality.to_string(index=False))
    print(f"Wrote validated data to {Path(args.output_dir).resolve()}")
    return 0 if quality["ok"].all() else 1


def list_factors_command(args: argparse.Namespace) -> int:
    names = FACTOR_REGISTRY.names(args.family)
    manifest = FACTOR_REGISTRY.manifest(names)
    if manifest.empty:
        available = ", ".join(FACTOR_REGISTRY.families())
        raise SystemExit(f"Unknown family {args.family!r}. Available families: {available}")
    print(manifest.to_string(index=False))
    print(f"\n{len(manifest)} registered factors")
    return 0


def build_factors_command(args: argparse.Namespace) -> int:
    universe = _argument_symbols(args)
    frames = _fetch(args, include_benchmark=True)
    quality = pd.DataFrame(
        [validate_ohlcv(frame, symbol).as_record() for symbol, frame in frames.items()]
    )
    names = args.factors or FACTOR_REGISTRY.names()
    factors = compute_factor_zoo(frames, names, benchmark=args.benchmark)
    factors = {name: values.reindex(columns=universe) for name, values in factors.items()}
    write_data_outputs(frames, quality, args.output_dir)
    write_factor_outputs(factors, FACTOR_REGISTRY.manifest(names), args.output_dir)
    print(FACTOR_REGISTRY.manifest(names).to_string(index=False))
    print(f"Wrote {len(factors)} factors to {Path(args.output_dir).resolve()}")
    return 0


def smoke_backtest_command(args: argparse.Namespace) -> int:
    universe = _argument_symbols(args)
    frames = _fetch(args, include_benchmark=True)
    quality = pd.DataFrame(
        [validate_ohlcv(frame, symbol).as_record() for symbol, frame in frames.items()]
    )
    prices = close_matrix(frames)
    factor = compute_factor_zoo(
        frames,
        [args.factor],
        benchmark=args.benchmark,
    )[args.factor].reindex(columns=universe)
    factor = orient_factor(factor, args.factor)
    if args.invert_factor:
        factor = -factor
    weights = rank_target_weights(
        factor,
        prices,
        top_n=args.top_n,
        frequency=args.frequency,
        signal_lag=1,
        long_short=args.long_short,
    )
    weights = weights.reindex(columns=prices.columns, fill_value=0.0)
    artifacts = run_weight_backtest(
        prices,
        weights,
        name=args.factor,
        benchmark_symbol=args.benchmark,
        initial_capital=args.initial_capital,
        commission_bps=args.commission_bps,
    )
    write_data_outputs(frames, quality, args.output_dir)
    write_backtest_outputs(artifacts, args.output_dir)
    print(artifacts.stats.to_string())
    print(f"Wrote backtest artifacts to {Path(args.output_dir).resolve()}")
    return 0


def causal_screen_command(args: argparse.Namespace) -> int:
    universe = _argument_symbols(args)
    frames = _fetch(args, include_benchmark=True)
    quality = pd.DataFrame(
        [validate_ohlcv(frame, symbol).as_record() for symbol, frame in frames.items()]
    )
    config = ScreeningConfig(
        train_sessions=args.train_sessions,
        test_sessions=args.test_sessions,
        forward_horizon=args.forward_horizon,
        var_lags=args.var_lags,
        common_factors=args.common_factors,
        ridge_alpha=args.ridge_alpha,
        covariance_shrinkage=args.covariance_shrinkage,
        permutations=args.permutations,
        simulation_batch_size=args.simulation_batch_size,
        effect_horizons=tuple(args.effect_horizons),
        stationarity_p_threshold=args.stationarity_p_threshold,
        stationarity_required_fraction=args.stationarity_required_fraction,
        max_difference_order=args.max_difference_order,
        q_threshold=args.q_threshold,
        granger_p_threshold=args.granger_p_threshold,
        min_abs_rank_ic=args.min_abs_rank_ic,
        min_sign_consistency=args.min_sign_consistency,
        sensitivity_draws=args.sensitivity_draws,
        min_sensitivity_rho=args.min_sensitivity_rho,
        min_assets=args.min_assets,
        min_dates=args.min_dates,
        max_factors=args.max_factors,
        top_n=args.top_n,
        rebalance_frequency=args.frequency,
        long_short=args.long_short,
        seed=args.seed,
    )
    result = walk_forward_causal_screen(
        frames,
        universe,
        benchmark=args.benchmark,
        factor_names=args.factors,
        config=config,
        initial_capital=args.initial_capital,
        commission_bps=args.commission_bps,
    )
    result.run_config["data_source"] = frames.provenance.as_record()
    write_data_outputs(frames, quality, args.output_dir)
    write_walk_forward_outputs(result, args.output_dir)
    print("Selection frequency:")
    print(result.selection_summary.head(15).to_string(index=False))
    print("\nOut-of-sample folds:")
    print(result.folds.to_string(index=False))
    print("\nCombined out-of-sample backtest:")
    print(result.backtest.stats.to_string())
    print(f"Wrote causal screening artifacts to {Path(args.output_dir).resolve()}")
    return 0


def bivariate_rho_command(args: argparse.Namespace) -> int:
    universe = _argument_symbols(args)
    specification = FACTOR_REGISTRY.get(args.factor)
    frames = _fetch(args, include_benchmark=specification.requires_benchmark)
    if frames.provenance.adjustment != "backward_total_return":
        raise SystemExit(
            "bivariate rho analysis requires backward total-return prices with "
            "splits and cash distributions"
        )
    prices = close_matrix(frames).reindex(columns=universe)
    factor = compute_factor_zoo(
        frames,
        [args.factor],
        benchmark=args.benchmark,
    )[args.factor].reindex(columns=universe)
    analysis_start = pd.Timestamp(args.analysis_start)
    analysis_end = pd.Timestamp(args.analysis_end)
    if analysis_start > analysis_end:
        raise SystemExit("--analysis-start must be on or before --analysis-end")
    analysis_dates = prices.index[(prices.index >= analysis_start) & (prices.index <= analysis_end)]
    config = BivariateSensitivityConfig(
        var_lags=args.var_lags,
        horizons=tuple(args.effect_horizons),
        sensitivity_draws=args.sensitivity_draws,
        rho_limit=args.rho_limit,
        overturn_probability=args.overturn_probability,
        kernel_bandwidth=args.kernel_bandwidth,
        stationarity_p_threshold=args.stationarity_p_threshold,
        min_observations=args.min_observations,
        workers=args.workers,
        seed=args.seed,
    )
    result = estimate_bivariate_rho(
        prices,
        factor,
        config,
        analysis_dates=analysis_dates,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.rho_star.to_csv(output / "rho_star.csv")
    result.exclusions.to_csv(output / "excluded_symbols.csv", index=False)
    run_config = {
        "method": "notebook_bivariate_rho_v1",
        "factor": args.factor,
        "analysis_start": analysis_start.strftime("%Y-%m-%d"),
        "analysis_end": analysis_end.strftime("%Y-%m-%d"),
        "data_start": args.start,
        "data_end": args.end,
        "symbols": universe,
        "data_source": frames.provenance.as_record(),
        "sensitivity": asdict(config),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(result.rho_star.to_string())
    if not result.exclusions.empty:
        print("\nExcluded symbols:")
        print(result.exclusions.to_string(index=False))
    print(f"Wrote bivariate rho values to {(output / 'rho_star.csv').resolve()}")
    return 0


def build_option_features_command(args: argparse.Namespace) -> int:
    chain = load_option_chain(args.input, provider=args.option_provider)
    config = OptionFeatureConfig(
        near_dte=args.near_dte,
        far_dte=args.far_dte,
        maximum_dte_distance=args.maximum_dte_distance,
        volume_zscore_window=args.volume_zscore_window,
    )
    features = build_daily_option_features(chain, config).reset_index()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".parquet", ".pq"}:
        try:
            features.to_parquet(output, index=False)
        except ImportError as exc:
            raise SystemExit(
                "Parquet output requires the optional data dependency: pip install -e '.[data]'"
            ) from exc
    else:
        features.to_csv(output, index=False)
    print(
        f"Wrote {len(features)} daily option-feature rows from {chain.provider} "
        f"to {output.resolve()}"
    )
    return 0


def acquire_free_dataset_command(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    if args.symbols_file is None:
        snapshot = fetch_nasdaq100_snapshot()
        universe = snapshot.constituents
        symbols = snapshot.symbols
        universe_basis = "current_nasdaq100_snapshot"
    else:
        source = pd.read_csv(args.symbols_file)
        if "symbol" not in source:
            raise SystemExit(f"{args.symbols_file} must contain a 'symbol' column")
        symbols = _symbols(source["symbol"].dropna().astype(str).tolist())
        if not symbols:
            raise SystemExit(f"{args.symbols_file} contains no symbols")
        universe = pd.DataFrame({"symbol": symbols})
        universe_basis = "provided_symbol_file"

    if args.provider == "polygon":
        market_source: MarketDataSource = PolygonMarketDataSource(
            cache_dir=args.cache_dir,
            api_key=args.api_key,
        )
    else:
        market_source = YahooFinanceSource(cache_dir=args.cache_dir)
    downloaded_prices = market_source.fetch_many(
        symbols,
        args.start,
        args.end,
        refresh=args.refresh,
    )
    prices, history_eligibility = select_price_history_eligible(
        downloaded_prices,
        requested_start=args.start,
        minimum_sessions=args.min_price_sessions,
        maximum_start_gap_days=args.max_start_gap_days,
    )
    universe = universe.merge(history_eligibility, on="symbol", how="left", validate="one_to_one")
    symbols = list(prices)
    quality = pd.DataFrame(
        [validate_ohlcv(frame, symbol).as_record() for symbol, frame in prices.items()]
    )
    write_data_outputs(prices, quality, output)

    option_detail = CboeOptionVolumeSource(
        cache_dir=args.option_cache_dir
    ).fetch_range(
        symbols,
        args.start,
        args.end,
        refresh=args.refresh,
    )
    option_daily = aggregate_cboe_option_volume(option_detail)
    universe["has_cboe_option_volume"] = universe["symbol"].isin(option_detail["Symbol"])
    archive = args.archive or output / "free_nasdaq100_bundle.tar.gz"
    archive_path = write_free_dataset_bundle(
        output,
        universe=universe,
        option_detail=option_detail,
        option_daily=option_daily,
        start_date=args.start,
        end_date=args.end,
        universe_basis=universe_basis,
        minimum_price_sessions=args.min_price_sessions,
        maximum_start_gap_days=args.max_start_gap_days,
        archive=archive,
        price_provenance=market_source.provenance,
    )
    print(
        f"Wrote {len(prices)} stock series and {len(option_daily)} daily Cboe "
        f"option-volume rows to {output.resolve()}"
    )
    print(f"Transfer archive: {archive_path.resolve()}")
    return 0 if quality["ok"].all() else 1


def mine_factors_command(args: argparse.Namespace) -> int:
    """Evolve new factor formulas via genetic programming (gplearn)."""
    from .factor_miner import (
        MiningConfig,
        compute_existing_factors,
        load_panel_data,
        mine_factors,
        save_mined_factors,
    )

    data_dir = Path(args.data_dir)
    ohlcv_path = data_dir / args.ohlcv_file
    if not ohlcv_path.exists():
        raise SystemExit(f"OHLCV file not found: {ohlcv_path}")

    option_volume_path: str | None = None
    if args.option_volume_file:
        opt_path = data_dir / args.option_volume_file
        if opt_path.exists():
            option_volume_path = str(opt_path)

    panel = load_panel_data(
        str(ohlcv_path),
        option_volume_path=option_volume_path,
        start_date=args.start,
        end_date=args.end,
    )
    print(
        f"Loaded panel: {panel['close'].shape[0]} dates x {panel['close'].shape[1]} symbols"
    )

    existing: dict | None = None
    if not args.no_existing_factors:
        existing = compute_existing_factors(panel)
        print(f"Computed {len(existing)} existing factor-zoo factors as terminals")

    config = MiningConfig(
        population_size=args.population,
        generations=args.generations,
        tournament_size=args.tournament_size,
        n_factors=args.n_factors,
        rankic_threshold=args.rankic_threshold,
        correlation_threshold=args.correlation_threshold,
        forward_return_horizon=args.forward_horizon,
        parsimony_coefficient=args.parsimony,
        random_state=args.seed,
    )

    mined = mine_factors(panel, config, existing_factors=existing)

    if not mined:
        print("No factors met the RankIC / correlation thresholds; try lowering them.")
        return 1

    csv_path = save_mined_factors(mined, args.output_dir, config)
    print(f"\nMined {len(mined)} factors:")
    for mf in mined:
        print(f"  RankIC={mf.rankic:+.4f}  IR={mf.rankic_ir:+.3f}  {mf.formula}")
    print(f"\nSaved summary to {csv_path.resolve()}")
    print(f"Saved programs to {Path(args.output_dir).resolve() / 'mined_programs.pkl'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="us-factor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--symbols", nargs="+", default=["EWY", "SPY", "QQQ", "IWM"])
        command.add_argument(
            "--symbols-file",
            type=Path,
            help="CSV universe with a symbol column; overrides --symbols",
        )
        command.add_argument("--start", default="2021-01-01")
        command.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
        command.add_argument(
            "--provider",
            choices=["polygon", "yahoo", "alpaca", "frozen"],
            default="polygon",
        )
        command.add_argument(
            "--interval",
            choices=["1d", "1m", "5m"],
            default="1d",
            help="bar interval for polygon provider",
        )
        command.add_argument("--api-key", default=None)
        command.add_argument("--cache-dir", default=".cache/market_data")
        command.add_argument("--source", type=Path)
        command.add_argument("--source-provider")
        command.add_argument("--source-feed")
        command.add_argument("--source-adjustment")
        command.add_argument("--source-session", default="regular")
        command.add_argument("--alpaca-feed", choices=["iex", "sip"], default="iex")
        command.add_argument(
            "--alpaca-adjustment",
            choices=["all", "split", "dividend", "raw"],
            default="all",
        )
        command.add_argument("--output-dir", default="results/smoke")
        command.add_argument("--refresh", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate-data", help="Fetch and validate one uniform daily OHLCV panel"
    )
    add_data_arguments(validate_parser)
    validate_parser.set_defaults(func=validate_data_command)

    list_parser = subparsers.add_parser(
        "list-factors", help="List registered factors and their metadata"
    )
    list_parser.add_argument("--family", choices=FACTOR_REGISTRY.families(), default=None)
    list_parser.set_defaults(func=list_factors_command)

    factor_parser = subparsers.add_parser(
        "build-factors", help="Compute and export registered factors"
    )
    add_data_arguments(factor_parser)
    factor_parser.add_argument("--factors", nargs="+", choices=FACTOR_REGISTRY.names())
    factor_parser.add_argument("--benchmark", default="SPY")
    factor_parser.set_defaults(func=build_factors_command)

    backtest_parser = subparsers.add_parser(
        "smoke-backtest", help="Run a cross-sectional registered-factor smoke test"
    )
    add_data_arguments(backtest_parser)
    backtest_parser.add_argument(
        "--factor", choices=FACTOR_REGISTRY.names(), default="momentum_63d"
    )
    backtest_parser.add_argument("--invert-factor", action="store_true")
    backtest_parser.add_argument("--top-n", type=int, default=2)
    backtest_parser.add_argument(
        "--frequency", choices=["daily", "weekly", "monthly", "quarterly"], default="monthly"
    )
    backtest_parser.add_argument("--long-short", action="store_true")
    backtest_parser.add_argument("--benchmark", default="SPY")
    backtest_parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    backtest_parser.add_argument("--commission-bps", type=float, default=5.0)
    backtest_parser.set_defaults(func=smoke_backtest_command)

    causal_parser = subparsers.add_parser(
        "causal-screen",
        help="Run conditional factor screening with walk-forward validation",
    )
    add_data_arguments(causal_parser)
    causal_parser.set_defaults(
        symbols=DEFAULT_VALIDATION_UNIVERSE,
        start="2021-01-04",
        end="2025-12-31",
        output_dir="results/causal_screen",
    )
    causal_parser.add_argument("--benchmark", default="SPY")
    causal_parser.add_argument("--factors", nargs="+", choices=FACTOR_REGISTRY.names())
    causal_parser.add_argument("--train-sessions", type=int, default=252)
    causal_parser.add_argument("--test-sessions", type=int, default=63)
    causal_parser.add_argument("--forward-horizon", type=int, default=5)
    causal_parser.add_argument("--var-lags", type=int, default=7)
    causal_parser.add_argument("--common-factors", type=int, default=3)
    causal_parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    causal_parser.add_argument("--covariance-shrinkage", type=float, default=0.25)
    causal_parser.add_argument("--permutations", type=int, default=999)
    causal_parser.add_argument("--simulation-batch-size", type=int, default=8)
    causal_parser.add_argument(
        "--effect-horizons",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7],
    )
    causal_parser.add_argument("--stationarity-p-threshold", type=float, default=0.05)
    causal_parser.add_argument("--stationarity-required-fraction", type=float, default=0.80)
    causal_parser.add_argument("--max-difference-order", type=int, choices=[0, 1], default=1)
    causal_parser.add_argument("--q-threshold", type=float, default=0.10)
    causal_parser.add_argument("--granger-p-threshold", type=float, default=0.05)
    causal_parser.add_argument("--min-abs-rank-ic", type=float, default=0.01)
    causal_parser.add_argument("--min-sign-consistency", type=float, default=0.52)
    causal_parser.add_argument("--sensitivity-draws", type=int, default=999)
    causal_parser.add_argument("--min-sensitivity-rho", type=float, default=0.10)
    causal_parser.add_argument("--min-assets", type=int, default=8)
    causal_parser.add_argument("--min-dates", type=int, default=126)
    causal_parser.add_argument("--max-factors", type=int, default=5)
    causal_parser.add_argument("--top-n", type=int, default=5)
    causal_parser.add_argument(
        "--frequency",
        choices=["daily", "weekly", "monthly", "quarterly"],
        default="weekly",
    )
    causal_parser.add_argument("--long-short", action="store_true")
    causal_parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    causal_parser.add_argument("--commission-bps", type=float, default=5.0)
    causal_parser.add_argument("--seed", type=int, default=7)
    causal_parser.set_defaults(func=causal_screen_command)

    bivariate_parser = subparsers.add_parser(
        "bivariate-rho",
        help="Estimate notebook-style sensitivity rho for one factor and each symbol",
    )
    add_data_arguments(bivariate_parser)
    bivariate_parser.set_defaults(
        symbols=DEFAULT_VALIDATION_UNIVERSE,
        start="2023-09-01",
        end="2026-12-31",
        output_dir="results/bivariate_momentum_2024_2026",
    )
    bivariate_parser.add_argument(
        "--factor",
        choices=FACTOR_REGISTRY.names(),
        default="momentum_63d",
    )
    bivariate_parser.add_argument("--benchmark", default="SPY")
    bivariate_parser.add_argument("--analysis-start", default="2024-01-01")
    bivariate_parser.add_argument("--analysis-end", default="2026-12-31")
    bivariate_parser.add_argument("--var-lags", type=int, default=7)
    bivariate_parser.add_argument(
        "--effect-horizons",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7],
    )
    bivariate_parser.add_argument("--sensitivity-draws", type=int, default=999)
    bivariate_parser.add_argument("--rho-limit", type=float, default=0.99)
    bivariate_parser.add_argument("--overturn-probability", type=float, default=0.05)
    bivariate_parser.add_argument("--kernel-bandwidth", type=float, default=0.08)
    bivariate_parser.add_argument("--stationarity-p-threshold", type=float, default=0.05)
    bivariate_parser.add_argument("--min-observations", type=int, default=252)
    bivariate_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="worker processes; 0 reserves eight CPUs automatically",
    )
    bivariate_parser.add_argument("--seed", type=int, default=7)
    bivariate_parser.set_defaults(func=bivariate_rho_command)

    option_parser = subparsers.add_parser(
        "build-option-features",
        help="Validate normalized option observations and build daily features",
    )
    option_parser.add_argument("--input", type=Path, required=True)
    option_parser.add_argument("--output", type=Path, required=True)
    option_parser.add_argument("--option-provider")
    option_parser.add_argument("--near-dte", type=int, default=30)
    option_parser.add_argument("--far-dte", type=int, default=90)
    option_parser.add_argument("--maximum-dte-distance", type=int, default=21)
    option_parser.add_argument("--volume-zscore-window", type=int, default=20)
    option_parser.set_defaults(func=build_option_features_command)

    free_data_parser = subparsers.add_parser(
        "acquire-free-data",
        help="Build a portable Nasdaq-100 OHLCV and Cboe option-volume bundle",
    )
    free_data_parser.add_argument("--start", default="2024-01-01")
    free_data_parser.add_argument(
        "--end",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    free_data_parser.add_argument(
        "--symbols-file",
        type=Path,
        help=(
            "optional CSV with a symbol column; otherwise use the current official "
            "Nasdaq-100 snapshot"
        ),
    )
    free_data_parser.add_argument("--cache-dir", default=".cache/market_data")
    free_data_parser.add_argument(
        "--option-cache-dir",
        default=".cache/cboe_option_volume",
    )
    free_data_parser.add_argument(
        "--output-dir",
        default="data/free_nasdaq100_2024_2026",
    )
    free_data_parser.add_argument(
        "--min-price-sessions",
        type=int,
        default=252,
        help="exclude symbols with fewer requested-window price observations",
    )
    free_data_parser.add_argument(
        "--max-start-gap-days",
        type=int,
        default=10,
        help="exclude symbols whose price history starts this many days after --start",
    )
    free_data_parser.add_argument(
        "--archive",
        type=Path,
        help="output .tar.gz path; defaults inside --output-dir",
    )
    free_data_parser.add_argument("--refresh", action="store_true")
    free_data_parser.set_defaults(func=acquire_free_dataset_command)

    mine_parser = subparsers.add_parser(
        "mine-factors",
        help="Evolve new factor formulas via genetic programming (gplearn)",
    )
    mine_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/free_nasdaq100_2024_2026"),
        help="directory containing ohlcv.csv and (optional) cboe_option_volume_daily.csv.gz",
    )
    mine_parser.add_argument(
        "--ohlcv-file",
        default="ohlcv.csv",
        help="OHLCV CSV filename inside --data-dir",
    )
    mine_parser.add_argument(
        "--option-volume-file",
        default="cboe_option_volume_daily.csv.gz",
        help="option-volume CSV filename inside --data-dir (empty string to skip)",
    )
    mine_parser.add_argument("--start", default=None)
    mine_parser.add_argument("--end", default=None)
    mine_parser.add_argument("--population", type=int, default=1000)
    mine_parser.add_argument("--generations", type=int, default=20)
    mine_parser.add_argument("--tournament-size", type=int, default=20)
    mine_parser.add_argument("--n-factors", type=int, default=10)
    mine_parser.add_argument("--rankic-threshold", type=float, default=0.02)
    mine_parser.add_argument("--correlation-threshold", type=float, default=0.7)
    mine_parser.add_argument("--forward-horizon", type=int, default=1)
    mine_parser.add_argument("--parsimony", type=float, default=0.001)
    mine_parser.add_argument("--seed", type=int, default=42)
    mine_parser.add_argument(
        "--no-existing-factors",
        action="store_true",
        help="skip computing the 61 factor-zoo factors as terminals",
    )
    mine_parser.add_argument("--output-dir", default="results/mined_factors")
    mine_parser.set_defaults(func=mine_factors_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
