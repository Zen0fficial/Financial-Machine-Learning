"""US factor research foundations: market data, factor weights, and backtests."""

from .backtest import BacktestArtifacts, run_weight_backtest
from .bivariate_sensitivity import (
    BivariateRhoResult,
    BivariateSensitivityConfig,
    estimate_bivariate_rho,
)
from .causal_screening import (
    FactorScreeningResult,
    ScreeningConfig,
    WalkForwardResult,
    screen_factor_horizons,
    screen_factors,
    walk_forward_causal_screen,
)
from .data import (
    DataQualityReport,
    MarketDataPanel,
    MarketDataProvenance,
    MarketDataSource,
    YahooFinanceSource,
    validate_ohlcv,
)
from .factor_zoo import (
    FACTOR_REGISTRY,
    FactorRegistry,
    FactorSpec,
    compute_factor_zoo,
    orient_factor,
)
from .factors import momentum, rank_target_weights
from .free_dataset import (
    CboeOptionVolumeSource,
    Nasdaq100Snapshot,
    aggregate_cboe_option_volume,
    fetch_nasdaq100_snapshot,
    select_price_history_eligible,
    write_free_dataset_bundle,
)
from .fundamentals import (
    FUNDAMENTALS_REGISTRY,
    FundamentalsConfig,
    FundamentalsPanel,
    compute_fundamental_factors,
    orient_fundamental_factor,
)
from .options import (
    OptionChain,
    OptionFeatureConfig,
    OptionObservation,
    build_daily_option_features,
    make_option_supervised_frame,
)
from .providers import AlpacaMarketDataSource, FrozenMarketDataSource, PolygonMarketDataSource

__all__ = [
    "BacktestArtifacts",
    "BivariateRhoResult",
    "BivariateSensitivityConfig",
    "DataQualityReport",
    "FACTOR_REGISTRY",
    "AlpacaMarketDataSource",
    "CboeOptionVolumeSource",
    "FactorRegistry",
    "FactorScreeningResult",
    "FactorSpec",
    "FrozenMarketDataSource",
    "FUNDAMENTALS_REGISTRY",
    "FundamentalsConfig",
    "FundamentalsPanel",
    "MarketDataPanel",
    "MarketDataProvenance",
    "MarketDataSource",
    "OptionChain",
    "OptionFeatureConfig",
    "OptionObservation",
    "Nasdaq100Snapshot",
    "PolygonMarketDataSource",
    "ScreeningConfig",
    "YahooFinanceSource",
    "WalkForwardResult",
    "compute_factor_zoo",
    "aggregate_cboe_option_volume",
    "compute_fundamental_factors",
    "estimate_bivariate_rho",
    "fetch_nasdaq100_snapshot",
    "build_daily_option_features",
    "make_option_supervised_frame",
    "momentum",
    "orient_factor",
    "orient_fundamental_factor",
    "rank_target_weights",
    "run_weight_backtest",
    "screen_factors",
    "screen_factor_horizons",
    "select_price_history_eligible",
    "validate_ohlcv",
    "walk_forward_causal_screen",
    "write_free_dataset_bundle",
]
