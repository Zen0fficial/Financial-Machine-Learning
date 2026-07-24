# US Factor Screening

Research infrastructure for screening factors on US-listed stocks and ETFs. The project has four foundations:

1. Market data comes through a provenance-aware provider contract, with cached Yahoo Finance as the default.
2. Portfolio simulations run through the open-source [`bt`](https://github.com/pmorissette/bt) framework instead of custom accounting code.
3. A metadata-driven factor zoo supplies reusable, auditable OHLCV factors.
4. The `Analysis` notebook's Granger/Fisher screen controls false discoveries before walk-forward validation.

## What Works Now

- Inclusive start/end date Yahoo Finance OHLCV downloads
- Optional adjusted Alpaca daily bars from an explicit IEX or SIP feed
- Frozen CSV/Parquet snapshot replay with a required data-definition manifest
- Panel-level rejection of mixed providers, feeds, adjustments, intervals, or sessions
- Symbol validation, rate-limit retries, stale-data rejection, and a date-keyed cache
- A local, date-keyed CSV cache to make research runs repeatable
- Positive backward total-return Yahoo prices with splits and cash distributions embedded
- Strict schema and OHLCV invariant checks
- US-listed symbol scope, including `EWY`
- Cross-sectional trailing-momentum scores
- A 32-factor registry with formulas, lookbacks, input requirements, and conventional directions
- Reversal, momentum, trend, risk, distribution, liquidity, and microstructure families
- Compressed per-factor exports and coverage diagnostics
- Daily, weekly, monthly, or quarterly target weights
- One-observation signal lag to prevent same-close lookahead
- Long-only and market-neutral rank portfolios
- `bt` portfolio accounting, commissions, benchmark comparison, and performance statistics
- CSV artifacts for data, quality checks, weights, equity curves, and summary statistics
- One-year training and three-month out-of-sample walk-forward folds
- Reduced-rank multivariate Granger tests, conditional VAR randomization, sensitivity analysis, and Benjamini-Hochberg FDR control
- Notebook-faithful bivariate sensitivity screening with one rho value per stock and horizon
- Frozen out-of-sample factor signs and weights, with a forward-label embargo
- A strict normalized option schema and daily surface, skew, flow, and spread features

## Setup

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Any Python 3.11+ interpreter can be used. The project has no runtime or source-path dependency on TradingAgents. Its standalone Yahoo provider retains the useful behavioral ideas from that data pipeline: inclusive requested dates, adjusted prices, retry handling, stale-response rejection, normalization, validation, and caching.

Install the optional Parquet engine only when frozen Parquet input or output is needed:

```bash
.venv/bin/python -m pip install -e '.[dev,data]'
```

## Market Data Providers

Every research run now carries this data identity and an SHA-256 snapshot checksum into `market_data_manifest.json`:

```text
provider + feed + adjustment + interval + session
```

All symbols in one factor or backtest panel must have exactly the same identity. There is no per-symbol fallback from Alpaca to Yahoo because that would make cross-sectional prices and volumes incomparable.

Yahoo remains the backward total-return daily default. Its adjusted-close ratio is applied
backward to OHLC values and incorporates splits and cash distributions:

```bash
.venv/bin/us-factor validate-data \
  --provider yahoo \
  --symbols EWY SPY \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --output-dir results/data_check
```

Alpaca is an optional comparison source. The default is adjusted IEX data; select SIP only when the account is entitled to it. Alpaca documents `raw`, `split`, `dividend`, and `all` adjustments on its [historical bars endpoint](https://docs.alpaca.markets/us/v1.1/reference/stockbarsingle-1).

```bash
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."

.venv/bin/us-factor validate-data \
  --provider alpaca \
  --alpaca-feed iex \
  --alpaca-adjustment all \
  --symbols EWY SPY \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --output-dir results/alpaca_check
```

IEX is one exchange rather than the consolidated US tape, so IEX volume must not be compared with Yahoo or SIP volume inside one panel.

Outputs from new runs can be replayed directly because they contain a manifest. Replay fails if the snapshot content no longer matches its checksum:

```bash
.venv/bin/us-factor validate-data \
  --provider frozen \
  --source results/data_check \
  --symbols EWY SPY \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --output-dir results/frozen_replay
```

For an older or vendor CSV without a manifest, declare all three source fields explicitly with `--source-provider`, `--source-feed`, and `--source-adjustment`. Supported frozen layouts are a long `ohlcv.csv`/Parquet file with a `Symbol` column or per-symbol files such as `AAPL.csv` and `AAPL.parquet`.

## Build the Free Nasdaq-100 Data Bundle

Run the acquisition on a machine that can reach the public sources, then transfer the
resulting archive to the server:

```bash
.venv/bin/us-factor acquire-free-data \
  --start 2024-01-01 \
  --end 2026-07-22 \
  --output-dir data/free_nasdaq100_2024_2026
```

The command combines three free sources/contracts:

- the official Nasdaq endpoint for the current Nasdaq-100 constituent snapshot
- Yahoo Finance for daily backward total-return-adjusted OHLCV, including split and cash-dividend adjustment
- [Cboe historical statistics](https://www.cboe.com/us/options/market_statistics/historical_data/) for executed daily option volume on `CBOE`, `BATS`, `C2`, and `EDGX`

The default history rule excludes a symbol when its requested-window prices begin more
than 10 calendar days after `--start` or contain fewer than 252 sessions. Every source
symbol and exclusion reason remains in `nasdaq100_universe.csv`. This rule is only a
price-history proxy for recent listings; it does not measure free float.

The portable `free_nasdaq100_bundle.tar.gz` contains `ohlcv.csv`, data-quality and source
manifests, the universe record, venue/root-level option volume, and daily
underlying-level option volume. The free Cboe report is not a historical option chain:
it is Cboe-venue volume only and has no strike, expiration, call/put split, quotes, open
interest, implied volatility, or Greeks. IV, skew, and term-structure factors therefore
still require a separate contract-level archive.

After extracting the archive, the OHLCV portion can be replayed without network access:

```bash
.venv/bin/us-factor validate-data \
  --provider frozen \
  --source free_nasdaq100 \
  --symbols-file free_nasdaq100/nasdaq100_universe.csv \
  --start 2024-01-01 \
  --end 2026-07-22 \
  --output-dir results/server_data_check
```

## Validate EWY Data

```bash
.venv/bin/us-factor validate-data \
  --symbols EWY SPY \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --output-dir results/data_check
```

The command fails if required columns are absent, prices are nonpositive, volume is negative, values are missing, or an OHLC bar is internally inconsistent.

## Factor Zoo

List all registered factors, or filter by family:

```bash
.venv/bin/us-factor list-factors
.venv/bin/us-factor list-factors --family risk
```

Build the complete zoo from validated market data:

```bash
.venv/bin/us-factor build-factors \
  --symbols EWY SPY QQQ IWM \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --benchmark SPY \
  --output-dir results/factor_zoo
```

Use `--factors momentum_252_21d market_beta_60d` to build a subset. Outputs include `factor_manifest.csv`, `factor_coverage.csv`, and one compressed Date-by-Symbol matrix per factor under `factors/`.

| Family | Count | Examples |
|---|---:|---|
| Reversal | 3 | 1-day, 5-day, and 21-day reversal |
| Momentum | 6 | 1/3/6/12-month, 12-to-1-month, benchmark-relative |
| Trend | 5 | SMA distance, 252-day drawdown, RSI |
| Risk | 8 | Realized/downside/Parkinson/idiosyncratic volatility, beta, correlation |
| Distribution | 2 | Return skewness and maximum daily return |
| Liquidity | 5 | Dollar volume, Amihud, volume surprise/trend/pressure |
| Microstructure | 3 | Intraday return, overnight return, close location |

Factor files contain raw, interpretable measurements. The manifest's `conventional_long_direction` field records whether a portfolio should conventionally prefer higher or lower raw values. Factors marked `unspecified` are left unoriented so their sign can be established empirically.

## Run the Smoke Backtest

```bash
.venv/bin/us-factor smoke-backtest \
  --symbols EWY SPY QQQ IWM \
  --start 2021-01-01 \
  --end 2025-12-31 \
  --factor momentum_63d \
  --top-n 2 \
  --benchmark SPY \
  --commission-bps 5 \
  --output-dir results/etf_momentum
```

At each month end, the example holds the two ETFs with the highest conventionally oriented factor value. A one-row lag means the rebalance uses information from the previous trading close. `bt` then handles positions, cash, commissions, equity, drawdowns, and performance statistics. Use `--invert-factor` to test the opposite side of a hypothesis.

This is an infrastructure smoke test, not evidence that ETF momentum is a production-ready factor.

## Option Feature Boundary

Normalized option observations use timestamped contract-level fields for quotes, volume, open interest, underlying price, implied volatility, and Greeks. Daily aggregation currently produces put/call ratios, volume surprise, quoted spread, 30/90-day ATM volatility, term slope, and 25-delta skew.

```bash
.venv/bin/us-factor build-option-features \
  --input data/options_history.csv \
  --output results/options/daily_features.csv
```

Malformed rows, duplicate `(as_of, contract)` observations, and mixed option providers are rejected. End-of-day option features use at least a one-session execution lag when supervised targets are built. These features are not registered in the factor zoo yet: doing so requires enough timestamped history to cover the training, embargo, test, and untouched validation periods.

## Run One Bivariate Rho Experiment

The focused bivariate path estimates only the notebook sensitivity tipping point
`rho_star` for one factor, separately for every stock and forecast horizon. The default
experiment uses `momentum_63d`, analyzes 2024 through 2026, fits seven VAR lags, evaluates
horizons one through seven, and uses 999 sensitivity draws. Data starts in September 2023
to provide the 63-session signal warm-up.

Provide a CSV with a `symbol` column for a full Nasdaq-100 universe:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/us-factor bivariate-rho \
  --symbols-file /path/to/nasdaq_100.csv \
  --workers 56 \
  --output-dir results/bivariate_momentum_2024_2026
```

With `--workers 0`, the command automatically reserves eight CPUs, so a 64-CPU server
uses 56 worker processes. Each worker handles one stock and vectorizes its 999 sensitivity
paths. The command writes `rho_star.csv`, `excluded_symbols.csv`, and `run_config.json`.
It does not select factors or run a portfolio backtest. Nonpositive adjusted prices,
internal data gaps, nonstationary returns, persistently nonstationary signals, and unstable
VAR fits are excluded explicitly.

The output has one row per security:

```text
Symbol,rho_h1,rho_h2,rho_h3,rho_h4,rho_h5,rho_h6,rho_h7
AAPL,...
MSFT,...
```

## Run Causal Factor Screening

The default validation run screens all 32 factors on a diversified set of 16 liquid US stocks. Each fold uses 252 trading sessions for training, a five-session embargo matching the forward-return horizon, and 63 trading sessions for testing:

```bash
.venv/bin/us-factor causal-screen \
  --start 2021-01-04 \
  --end 2025-12-31 \
  --benchmark SPY \
  --permutations 999 \
  --commission-bps 5 \
  --output-dir results/causal_screen
```

The source notebook has one signal series and one EWY price series. A cross-sectional factor instead has one value per stock per date. The causal engine extends the notebook with a reduced-rank multivariate VAR. It learns return and signal PCA states inside each training fold, combines each stock's own lags with low-rank cross-stock lag maps, and tests the complete signal-to-return map as one hypothesis per factor rule. This avoids treating correlated stocks as independent evidence without trying to estimate an unidentifiable unrestricted VAR from one year of data.

For each factor and training fold, the screen:

1. Converts each factor to daily cross-sectional ranks and applies the ADF test separately to the log-price and factor-rank panels. A panel stays in levels when at least 80% of its asset series are stationary; otherwise it is first-differenced and tested again.
2. Fits training-only PCA bases for common return and signal states. Three components are used by default.
3. Calculates joint Granger SSR F-statistics at horizons 1 through 7 for direct own-stock lags and low-rank cross-stock signal paths. Every horizon conditions only on information available at the forecast origin, avoiding future-outcome leakage in the notebook's shifted regression.
4. Fits the multivariate VAR assignment model. Its innovation conditioning uses a shrinkage estimate of the full cross-stock return/signal residual covariance.
5. Simulates 999 complete factor panels from jointly sampled empirical residual vectors conditional on the realized return panel. The same paths provide the Fisher reference distribution for every horizon.
6. Applies Benjamini-Hochberg correction across the factor zoo separately at each horizon.
7. For statistically effective horizons, runs 999 confounding perturbations. `confounding_rho_star` is the smallest absolute treatment/outcome error correlation that makes that horizon insignificant.
8. Reports `robust_effect_horizon_sessions` as the last consecutively effective and confounding-robust horizon starting at day 1. Isolated later rejections are reported but do not extend the lifetime.
9. Requires significance, Fisher FDR, minimum rank IC, sign consistency, and confounding robustness at the configured trading horizon before selection.
10. Freezes selected factors' training signs and IC-based weights for the next 63 sessions, rebalances weekly with a one-session lag, and stitches non-overlapping test blocks into one `bt` simulation.

The output also records the selected difference order, ADF pass fractions, number of joint Granger restrictions, direct signal-effect norm, cross-asset signal-effect norm, and the share of assignment conditioning carried by off-diagonal cross-stock terms. `--effect-horizons`, `--stationarity-required-fraction`, `--common-factors`, `--ridge-alpha`, `--covariance-shrinkage`, and `--simulation-batch-size` expose the main controls. Setting `--common-factors 0` leaves only the direct own-stock channel; a one-asset, one-day input remains algebraically equivalent to the notebook's Granger test.

The screen produces:

- `screening_metrics.csv`: selection diagnostics at the configured trading horizon for every factor and fold
- `horizon_metrics.csv`: Granger, Fisher, FDR, rank-IC, stationarity, and confounding diagnostics at every tested horizon
- `selection_summary.csv`: selection count and average diagnostics by factor
- `walk_forward_folds.csv`: exact train, embargo, and test boundaries plus fold returns
- `walk_forward_target_weights.csv`: the frozen out-of-sample portfolio targets
- `run_config.json`: the complete reproducible statistical and portfolio configuration
- `backtest/`: combined out-of-sample equity, realized weights, and performance statistics

The word *causal* is deliberately qualified here. As in `Analysis`, interpreting a Fisher rejection causally requires unconfoundedness and a credible conditional assignment model. The sensitivity statistic measures how much treatment/outcome error correlation overturns a result, but it cannot prove that no omitted confounder exists. The screen is best used as a stricter signal-discovery filter, followed by untouched out-of-sample validation.

## Tests

```bash
.venv/bin/python -m pytest
```

The default suite is offline and deterministic. The opt-in live test exercises the standalone Yahoo provider with `EWY`:

```bash
RUN_LIVE_MARKET_TESTS=1 .venv/bin/python -m pytest -m integration -q
```

## Research Constraints

- Yahoo Finance is suitable for research prototyping, not an execution-grade market-data record.
- Alpaca IEX is a useful independent check, not a substitute for consolidated SIP volume.
- Frozen backtests require a single provider/feed/adjustment definition for the full panel.
- The current universe is a smoke-test universe, not a historical constituent database.
- The default liquid-stock validation universe is static and therefore has survivorship and universe-selection bias. Results are research validation, not investable evidence.
- Price signals use auto-adjusted Yahoo Finance history.
- The portfolio model currently includes proportional commissions but not spread, market impact, borrow cost, or capacity constraints.
- This zoo intentionally contains only factors derivable point-in-time from daily OHLCV. Historical fundamental families will require filing-date-aware data before they are backtested.
- Option feature definitions are implemented, but no option signal is eligible for screening until sufficient historical coverage is archived.

A production research process should add a point-in-time constituent universe, delisting returns, corporate-action auditing, sector/size neutralization, and spread and impact models.

## Method Reference

The conditional simulation, Fisher randomization test, and correlated-error sensitivity design are adapted from the local `Analysis` notebook and [Zhong, Huang, and Rubin (2025), *Fisher's Randomization Test for Causality with General Types of Treatments*](https://arxiv.org/abs/2501.06864). The reduced-rank multivariate formulation is this project's extension for applying one factor rule across a stock universe.
