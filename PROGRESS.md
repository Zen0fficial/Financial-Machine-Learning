# US Factor Screening: Current Progress

Last updated: 2026-07-22

## Objective

Build a standalone US equity research project that:

1. Maintains a reusable factor zoo.
2. Screens each factor rule with the causal procedure from the local `Analysis` notebook.
3. Controls false discoveries across the zoo.
4. Freezes signals after training and validates them with an out-of-sample `bt` backtest.

One factor is one trading rule, not one rule per stock. For example, `momentum_63d` scores every stock on a date and the portfolio buys the highest-ranked stocks. Some individual stocks may respond poorly; screening measures whether the rule is useful on average across all stock-date decisions, and backtesting measures whether the resulting portfolio makes money.

## Current Status

| Area | Status | Notes |
|---|---|---|
| Standalone market data | Complete | Yahoo default, optional Alpaca, frozen CSV/Parquet; no TradingAgents runtime dependency |
| Data provenance | Complete | Provider, feed, adjustment, interval, and session are uniform and recorded per panel |
| Data validation and cache | Complete | Inclusive dates, adjusted prices, retry backoff, atomic cache writes, stale-data guard, OHLCV checks |
| Free Nasdaq-100 bundle | Implemented, not fully downloaded | Current Nasdaq snapshot, Yahoo adjusted OHLCV, Cboe venue option volume, checksums, portable archive |
| Factor zoo | Complete | 32 daily OHLCV factors across seven families |
| Causal screening | Complete | Reduced-rank multivariate Granger statistic, conditional VAR, Fisher randomization, FDR (diagnostic only), sensitivity analysis |
| Bivariate rho pilot | Implemented, not run | One notebook-style rho per stock/horizon; parallel stock workers; 2024-2026 momentum defaults |
| Walk-forward design | Complete | 252 training sessions, 5-session embargo, 63 testing sessions |
| Backtesting | Complete | `bt`, weekly rebalance, one-session signal lag, commissions, SPY comparison |
| Reproducible artifacts | Complete | Metrics, folds, weights, equity, quality report, and JSON run configuration |
| Option feature boundary | Complete, dormant | Strict contract schema and daily flow/surface features; not in the zoo without adequate history |
| Initial live validation | Complete | No factor passed the predeclared causal/sensitivity gate |
| Production-ready universe | Not started | Current 16-stock universe is static and has survivorship bias |
| Historical fundamentals | Not started | Requires filing-date-aware, point-in-time data |

## Data Architecture

The project now owns its data contract in `src/us_factor_screening/data.py` and provider adapters in `src/us_factor_screening/providers.py`.

`YahooFinanceSource` directly uses the installed `yfinance` package and provides:

- US-listed symbol validation
- inclusive requested end dates despite Yahoo's exclusive `end` parameter
- auto-adjusted OHLCV history
- exponential backoff for Yahoo rate limits
- rejection of empty or stale responses
- deterministic `Date, Open, High, Low, Close, Volume` normalization
- OHLCV invariant validation
- a local date-keyed CSV cache

`AlpacaMarketDataSource` adds a credentialed US comparison feed with explicit `iex` or `sip` selection and explicit `raw`, `split`, `dividend`, or `all` corporate-action handling. `FrozenMarketDataSource` replays long or per-symbol CSV/Parquet snapshots. Every CLI data run returns a `MarketDataPanel`; its constructor rejects mixed provider definitions before factor construction or backtesting.

Every newly written dataset includes `market_data_manifest.json` with an SHA-256 snapshot checksum. Frozen replay verifies the checksum before loading. Older snapshots remain loadable when provider, feed, and adjustment are supplied explicitly. Yahoo provenance now names the price basis `backward_total_return`: the adjusted-close ratio embeds splits and cash distributions and is applied backward while the latest close remains on its current share basis.

Existing cache files remain compatible. TradingAgents is no longer imported, discovered, or required at runtime. The useful behavior of its Yahoo pipeline was reimplemented locally rather than retaining a repository dependency.

EWY remains only a liquid integration-test ticker. It is a US-listed South Korea ETF and is not part of the 16-stock US factor-validation universe.

## Free Nasdaq-100 Acquisition

`us-factor acquire-free-data` now creates a portable 2024-2026 research bundle without
TradingAgents or a paid data account. It uses the official current Nasdaq-100 snapshot,
Yahoo backward total-return-adjusted daily OHLCV, and Cboe's public historical daily
option-volume reports for its four US option venues.

The bundle contains the normalized OHLCV snapshot, quality and provenance manifests,
the full source universe with eligibility decisions, venue/root-level option volume,
underlying-level daily option volume, and SHA-256 checksums. Symbols lacking 252 sessions
or prices near the requested start are excluded from the downloaded research panel while
remaining visible in the universe record. This is a history-availability rule, not a
free-float measurement.

Cboe's free report is deliberately stored outside the contract-level option schema. It
does not provide consolidated OPRA volume, strikes, expirations, put/call identity,
quotes, open interest, IV, or Greeks. It can support option-activity factors only; the
surface and skew features remain dormant until a contract-level historical archive is
available.

## Option Feature Boundary

`src/us_factor_screening/options.py` adapts the normalized option schema and daily feature definitions from the MIT-licensed `Financial-Machine-Learning` repository without adding a runtime dependency. It provides:

- strict timestamp, quote, expiration, numeric, provider, and duplicate validation
- put/call volume and open-interest ratios
- option-volume surprise based only on prior observations
- quoted spread, 30/90-day ATM IV, term slope, and 25-delta skew
- conservative supervised targets whose return window begins after an execution lag

The feature code is ready for a frozen historical archive, but option factors are deliberately absent from the active zoo until the data spans enough training and testing regimes.

## Factor Zoo

The registry contains 32 point-in-time daily OHLCV factors:

| Family | Count | Examples |
|---|---:|---|
| Reversal | 3 | 1-day, 5-day, 21-day |
| Momentum | 6 | 1/3/6/12-month, 12-to-1-month, relative momentum |
| Trend | 5 | SMA distance, drawdown, RSI |
| Risk | 8 | Realized/downside/Parkinson/idiosyncratic volatility, beta, correlation |
| Distribution | 2 | Skewness, maximum daily return |
| Liquidity | 5 | Dollar volume, Amihud, volume surprise/trend/pressure |
| Microstructure | 3 | Intraday return, overnight return, close location |

Fundamental factors are deliberately excluded for now because the project does not yet have filing-date-aware historical fundamentals. Using today's restated fundamentals in old training windows would introduce lookahead bias.

## Bivariate Rho Pilot

The new `bivariate-rho` command preserves the notebook's bivariate modeling unit. For each
stock, it pairs the daily log change in positive backward total-return price with one factor
series, checks stationarity, fits a seven-lag bivariate VAR, and finds one sensitivity
tipping point at each horizon from one through seven. The result is a stock-by-horizon rho
matrix; p-values, factor selection, and portfolio construction are not outputs of this path.

The default experiment is `momentum_63d` over 2024 through available 2026 observations,
with a September 2023 data warm-up and 999 sensitivity draws. Stock calculations use
separate processes, while all sensitivity paths for one stock are vectorized. Automatic
worker selection reserves eight CPUs, giving 56 workers on the intended 64-CPU server.
This experiment has not been launched against the full Nasdaq-100 universe.

## Causal Screening Method

The local `Analysis` notebook tests one signal series against one ETF price series. The factor project extends the same method to a stock panel while still producing one hypothesis per factor rule.

For every factor and training fold:

1. Convert factor values to cross-sectional ranks.
2. Run panel ADF tests on log prices and factor ranks; first-difference only a panel that fails the configured stationarity fraction, then retest it.
3. Learn return and signal PCA states inside the training fold.
4. Calculate joint Granger SSR F-statistics over horizons 1 through 7 using only origin-time information.
5. Fit a reduced-rank multivariate VAR for price and factor changes.
6. Simulate 999 full factor panels from jointly sampled empirical residual vectors conditional on the realized return panel, using shrinkage covariance conditioning.
7. Reuse those paths across horizons to obtain Fisher p-values and apply Benjamini-Hochberg correction across factors at each horizon.
8. For qualifying factor-horizons, run 999 correlated-error perturbations and estimate the minimum confounding correlation that overturns significance.
9. Define factor lifetime as the last consecutively effective and robust horizon beginning at day 1.

The default selection gate requires:

- Granger `p <= 0.05`
- raw Fisher randomization `p <= 0.05`
- absolute five-session rank IC of at least `0.01`
- IC sign consistency of at least `0.52`
- sensitivity robustness (`rho_star`) of at least `0.10`

The Benjamini-Hochberg FDR-adjusted `q` is computed and reported as a diagnostic
across factors at each horizon, but it is not a selection gate. The focus of the
procedure is the confounding sensitivity analysis, so selection ranks surviving
factor-horizons by `rho_star` then by absolute rank IC.

This is evidence of conditional predictive content under the assignment model. A causal interpretation still requires unconfoundedness; the method cannot prove that omitted confounders do not exist.

## Backtest Design

Each non-overlapping fold uses:

- 252 trading sessions for training
- a 5-session embargo matching the forward-return target
- 63 trading sessions for testing
- factor signs and IC-based factor weights frozen after training
- weekly portfolio rebalancing
- one-session lag before a close-derived signal can trade
- five equal-weight long positions by default
- 5 basis points proportional commission
- SPY as benchmark only, never as a ranked candidate

If no factor passes, the strategy holds cash. It does not fall back to the best-looking rejected factor.

## Legacy Validation Baseline

The following result predates the multivariate horizon engine and used 499 Fisher draws. It remains a reproducible baseline, but it must not be presented as validation of the current 999-draw stationarity-aware method.

Universe:

`AAPL MSFT NVDA AMZN GOOGL META JPM BAC XOM CVX JNJ UNH PG KO HD CAT`

Benchmark: `SPY`

Period: `2021-01-04` through `2025-12-31`

Results:

- 1,255 aligned sessions for every symbol
- 11 out-of-sample test quarters
- 32 factors per fold
- 352 fold-factor tests
- 499 Fisher simulations per test
- 14 raw Fisher p-values at or below 5%
- 17 Granger p-values at or below 5%
- 6 tests passing both raw thresholds
- 0 tests surviving the full selection gate

The strongest complete near-miss was fold-10 `volume_surprise_20d`:

- mean rank IC: `0.056`
- Granger p-value: `0.013`
- raw Fisher p-value: `0.008`
- FDR-adjusted Fisher q-value: `0.256`

The adjusted q-value means the raw result was not strong enough after searching all 32 factors. It is not a 25.6% probability that the factor is false.

Because no factor qualified, all test-period target weights were zero:

- causal-screened strategy return: `0.00%`
- SPY return over the stitched test period: `74.35%`

This is a valid negative research result. Relaxing thresholds after seeing it would convert the validation into data mining.

Artifacts are under `results/causal_screen_rolling_2021_2025/`:

- `screening_metrics.csv`
- `selection_summary.csv`
- `walk_forward_folds.csv`
- `walk_forward_target_weights.csv`
- `run_config.json`
- `backtest/summary.csv`
- `backtest/equity_curve.csv`
- `backtest/realized_weights.csv`

## Verification

- Offline suite: 49 tests passed, including free-source parsing/bundling, multivariate cross-asset recovery, null-signal coverage, adaptive stationarity transforms, and three-session effect-decay recovery
- Frozen replay smoke test: AAPL and SPY each reproduced 1,255 validated sessions and wrote a uniform Yahoo-adjusted manifest
- Full cached panel: all 17 symbols and 1,255 sessions validated through the standalone source
- Forced live EWY refresh: reached Yahoo directly but was rejected with HTTP 429 after all retries on 2026-07-22
- Free-source smoke check: official Nasdaq snapshot returned 103 securities dated 2026-07-21; Cboe returned 23 EWY venue rows and 465 contracts for 2024-01-02 through 2024-01-10
- Granger equivalence test: the single-series implementation matches `statsmodels.grangercausalitytests`
- Lint: clean
- Dependency check: no broken requirements

Provider tests cover inclusive dates, caching, stale responses, exponential rate-limit retries, explicit Alpaca feed/adjustment parameters, mixed-panel rejection, and frozen snapshot replay. Option tests cover schema validation, duplicate rejection, surface and flow features, prior-only volume baselines, and execution-lagged targets. The current Yahoo live-refresh failure is external vendor state, not a TradingAgents fallback or import. Cached research remains reproducible, while `--refresh` deliberately fails clearly instead of silently claiming cached data is current.

## Commands

Setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Validate data:

```bash
.venv/bin/us-factor validate-data \
  --symbols EWY SPY \
  --start 2021-01-04 \
  --end 2025-12-31 \
  --output-dir results/data_check
```

Build a portable free Nasdaq-100 dataset locally:

```bash
.venv/bin/us-factor acquire-free-data \
  --start 2024-01-01 \
  --end 2026-07-22 \
  --output-dir data/free_nasdaq100_2024_2026
```

Inspect the zoo:

```bash
.venv/bin/us-factor list-factors
```

Run the full rolling study:

```bash
.venv/bin/us-factor causal-screen \
  --start 2021-01-04 \
  --end 2025-12-31 \
  --benchmark SPY \
  --permutations 999 \
  --commission-bps 5 \
  --output-dir results/causal_screen
```

Run tests:

```bash
.venv/bin/python -m pytest -q
RUN_LIVE_MARKET_TESTS=1 .venv/bin/python -m pytest -m integration -q
```

## Known Limitations

1. The 16-stock universe is static, small, and selected using companies known today.
2. Yahoo Finance is suitable for prototyping, not an execution-grade data record.
3. The backtest has commissions but no spread, market impact, borrow cost, or capacity model.
4. There are no delisting returns or historical constituent changes.
5. Sector and size exposure are not neutralized.
6. Some zoo entries can be redundant after cross-sectional ranking; the FDR family does not yet cluster equivalent factors.
7. Fundamental, analyst-estimate, and alternative-data factors are not present.
8. The free Cboe option feed is venue-level aggregate volume, not consolidated or contract-level history.
9. The current Nasdaq-100 snapshot introduces survivorship bias for earlier dates.

## Recommended Next Milestones

1. Replace the static universe with a point-in-time US constituent universe including delisted securities.
2. Detect and group factors that are identical or near-identical after cross-sectional ranking before FDR correction.
3. Add sector and size neutralization, then repeat the predeclared study without changing significance thresholds.
4. Add filing-date-aware fundamental data and fundamental factor families.
5. Add spread, impact, borrow, and capacity models.
6. Maintain a final untouched holdout period that is evaluated only after the research design is frozen.

The next statistical step should improve the data and universe, not lower the causal or FDR thresholds.
