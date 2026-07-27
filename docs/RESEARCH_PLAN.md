# US Factor Research Plan

This project should run one shared research pipeline with different inputs for
Nasdaq-100 and S&P 500. The code path, causal model, covariate rules, signal
acceptance thresholds, strategy optimizer, and validation protocol must be the
same across both universes; only the universe file, market-data panel, benchmark,
metadata inputs, and output run id change.

The goal is to discover long-only strategies that improve out-of-sample Sharpe
ratio without turning the backtest into the research target. Signal discovery
and strategy selection use 2024-2025 training and walk-forward validation.
This-year performance is evaluated only after the signal library and strategy
configuration are frozen.

## Scope

- In: one parameterized pipeline for Nasdaq-100 and S&P 500 runs.
- In: daily OHLCV, point-in-time fundamentals when available, option-derived
  daily features when available, universe membership, benchmark, and metadata.
- In: conditional causal screening, sensitivity analysis, conditional novelty
  testing, AI signal proposals, long-only portfolio construction, and backtests.
- Out: separate universe-specific code paths, shorting, leverage, live trading,
  intraday execution, arbitrary agent-generated executable code, and tuning on
  the sealed holdout.

## Shared Pipeline Inputs

Each run supplies a small immutable run specification:

| Field | Nasdaq-100 run | S&P 500 run | Notes |
| --- | --- | --- | --- |
| run_id | nasdaq100 | sp500 | Controls output path only. |
| universe_file | Nasdaq-100 membership | S&P 500 membership | Must be date-aware when possible. |
| market_data | Nasdaq-100 OHLCV panel | S&P 500 OHLCV panel | Same normalized schema. |
| benchmark | QQQ | SPY | Benchmark is not a ranked candidate. |
| sector_metadata | optional | optional | Used only if point-in-time or stable enough. |
| option_features | optional | optional | Daily Date x Symbol features. |
| fundamentals | optional | optional | Point-in-time daily aligned fields. |
| output_dir | results/nasdaq100/ | results/sp500/ | Artifacts remain separate. |

Overlap between Nasdaq-100 and S&P 500 constituents is allowed. A stock that is
eligible in both universes is evaluated independently in each run because its
cross-section, benchmark, covariate distribution, and portfolio context differ.

## Action Items

[ ] Add a run-spec abstraction that parameterizes the existing CLI around run
id, symbols file, data source, benchmark, optional metadata paths, and output
directory.

[ ] Add a covariate registry in src/us_factor_screening/covariates.py, modeled
after the factor registry but with timing metadata: name, family, formula,
lookback, required inputs, availability lag, panel shape, and whether the
covariate is asset-level, universe-level, sector-level, or benchmark-level.

[ ] Build covariates from the current schema without changing the data-provider
contract. The OHLCV source remains normalized per symbol as Date, Open, High,
Low, Close, and Volume; FactorContext already converts this into Date x Symbol
matrices, which is the right base object for covariate construction.

[ ] Extend ScreeningConfig with covariate controls: selected covariate names,
maximum covariate components, lag count, ridge penalty, stationarity policy,
conditional novelty threshold, and whether accepted-signal residual components
are included for incremental tests.

[ ] Extend screen_factor_horizons and _analyze_factor to accept a covariate
panel bundle aligned to the same training dates and candidate symbols as the
candidate signal.

[ ] Replace the current return-signal VAR internals with a conditional
return-signal-covariate model. The return equation and signal-assignment
equation must include lagged covariate components while Fisher randomization
keeps the realized covariate history fixed.

[ ] Add conditional novelty testing. A candidate must be weakly correlated with
the existing signal library after both candidate and existing signals are
residualized on the same lagged covariates using training-only data.

[ ] Accept a signal only when it passes coverage, stationarity or differencing,
conditional rho, incremental rho, sign stability, effect-lifetime, and
conditional novelty gates. P-values and FDR-adjusted values remain diagnostics,
not the ranking target.

[ ] Run AI signal generation through the same registry and gate. The agent may
propose formulas from approved operators and source fields, but the pipeline
must store the proposal, rationale, source inputs, formula hash, diagnostics,
and rejection reason.

[ ] Build long-only strategies from accepted signals separately for each run:
equal-weight baseline, inverse-volatility baseline, rho-weighted signal
ensemble, and constrained risk-optimized ensemble.

[ ] Write artifacts separately under results/{run_id}/: covariate manifest,
signal manifest, causal metrics, rho horizons, novelty matrix, selected signal
library, target weights, realized weights, turnover, equity curve, benchmark
comparison, and run configuration.

[ ] Add tests for covariate timing, shape alignment, conditional-confounder
recovery, redundant-signal rejection, incremental-signal acceptance, universe
isolation, benchmark exclusion, embargo enforcement, and holdout isolation.

## Current Data Schema

The current daily market schema is already sufficient for a first covariate
layer:

- MarketDataPanel stores one normalized DataFrame per symbol and rejects mixed
  provider/feed/adjustment definitions.
- Each normalized OHLCV frame has Date, Open, High, Low, Close, and Volume.
- FactorContext.from_frames converts the symbol frames into Date x Symbol
  matrices: open, high, low, close, volume, plus derived returns, dollar
  volume, and benchmark returns.
- options.py converts normalized contract observations into daily option-feature
  rows by trade date and underlying.
- fundamentals.py aligns quarterly financial statements to the daily calendar
  by availability date, so fundamental covariates can be point-in-time when
  the input files are available.

The covariate layer should not create a new raw data schema. It should consume
these already-normalized panels and return one aligned object:

- asset_covariates: a mapping from covariate name to Date x Symbol DataFrame.
- universe_covariates: a mapping from covariate name to Date Series.
- benchmark_covariates: a mapping from covariate name to Date Series.
- sector_covariates: a mapping from covariate name to Date x Symbol DataFrame.
- manifest: a DataFrame with name, formula, source, lag, lookback, and role.

Before modeling, all covariates are expanded to a common three-dimensional
array:

    Z[t, i, k] = covariate k observed for asset i at date t

Universe-level and benchmark-level series are broadcast across assets. Sector
covariates are mapped onto each asset sector and then represented as Date x
Symbol panels. The model should also keep the lower-dimensional source panels
for diagnostics so a user can see whether the signal is being explained away by
market, sector, volatility, liquidity, or other accepted signals.

## Initial Covariate Set

The first implementation should start with covariates that are available from
OHLCV and do not require new data:

| Family | Covariate | Shape | Source | Timing rule |
| --- | --- | --- | --- | --- |
| Own return | lagged 1d, 5d, 21d returns | Date x Symbol | close | lag at least one session |
| Benchmark | lagged benchmark returns | Date series | benchmark close | lag at least one session |
| Volatility | 21d and 63d realized vol | Date x Symbol | returns | rolling window ending before trade |
| Liquidity | log dollar volume, turnover proxy, Amihud | Date x Symbol | close and volume | rolling window ending before trade |
| Breadth | universe mean return, percent above 50d SMA | Date series | eligible universe panel | computed within run only |
| Common risk | training-only return PCs | Date x Component | returns | fit PCA only on training fold |
| Trend state | benchmark 63d return and drawdown | Date series | benchmark close | lag at least one session |

Optional covariates can be added once the data is present:

| Family | Covariate | Shape | Source | Timing rule |
| --- | --- | --- | --- | --- |
| Sector | sector return and sector volatility | Date x Symbol | sector metadata plus OHLCV | no future sector membership |
| Options | option-volume surprise, put/call ratio, IV slope/skew | Date x Symbol | options.py features | use only prior close or later |
| Fundamentals | size, value, profitability, leverage, growth | Date x Symbol | fundamentals.py | availability-date aligned |

Covariates are not automatically safe. A covariate can be a confounder, a
mediator, or a collider depending on the candidate signal. The registry should
therefore include exclusion tags. For example, if a candidate signal directly
uses same-day volume, the model must not condition on same-day volume-derived
covariates; only lagged volume state is admissible.

## Incorporating Covariates Into Causal Modeling

The current causal screen uses two panels:

    Y[t, i] = stationary log-return outcome for asset i
    S[t, i] = stationary ranked candidate signal for asset i

It fits a reduced-rank multivariate VAR where lagged Y and lagged S predict both
future Y and future S. Fisher randomization simulates alternative signal
histories from the fitted assignment model conditional on realized returns.
Sensitivity analysis perturbs treatment and outcome errors to find the rho
value that overturns the effect.

The covariate extension should use:

    Y[t, i]      = stationary log-return outcome
    S[t, i]      = stationary ranked candidate signal
    Z[t, i, k]   = stationary covariate panel
    A[t, i, m]   = optional accepted-signal residual component panel

The restricted and unrestricted return tests become:

    Restricted:   Y[t+h, i] is explained by asset fixed effects, lags of Y,
                  lags of Z, and optional lags of accepted-signal components A.
    Unrestricted: the same model plus lags of candidate signal S.

The assignment model for Fisher randomization becomes:

    S[t, i] is explained by asset fixed effects, lags of S, lags of Y,
    lags of Z, and optional lags of A.

During Fisher randomization:

- realized Y is fixed,
- realized Z is fixed,
- accepted-signal components A are fixed for the incremental test,
- simulated S-star paths are drawn from the conditional assignment residuals,
- every horizon reuses the same simulated paths where possible.

During sensitivity analysis:

- the same conditional model is used,
- the outcome residual is computed from the covariate-conditioned restricted
  outcome model,
- rho measures the treatment/outcome residual correlation needed to overturn
  the candidate after controlling for covariates and existing signals.

This makes rho_star an incremental robustness measure: the candidate must
remain influential after accounting for market state, common risk, liquidity,
volatility, and the already-accepted signal library.

## Conditional Novelty

Signal novelty should be tested after removing covariate effects, not by raw
correlation. For each training fold:

1. Align candidate signal S, existing signal library L_j, and covariates Z on
   training dates and symbols.
2. Build lagged covariate design matrices using only information available at
   the signal date.
3. Fit training-only residualizers with cross-fitting when sample size allows:
   S on Z and L_j on Z.
4. Compute cross-sectional rank correlations between residualized S and each
   residualized L_j by date.
5. Store max absolute conditional correlation, median absolute conditional
   correlation, and the closest existing signal.

The initial rejection rule should be:

    reject if max_abs_conditional_corr > 0.30

If several candidates are correlated with each other but useful, keep the one
with stronger incremental rho_star, better coverage, lower turnover, and clearer
economic rationale. The rejected candidates remain in the ledger with their
nearest-neighbor signal and correlation value.

## Implementation Insertion Points

- factor_zoo.py: keep signal computation as-is; do not turn covariates into
  tradable factors by default.
- covariates.py: add covariate specs, builders, shape alignment, stationarity
  transforms, and manifest output.
- causal_screening.py: extend ScreeningConfig, FactorAnalysis, MultivariateVAR,
  Granger design construction, Fisher simulation, and sensitivity routines to
  accept covariate components.
- cli.py: add shared run-research and lower-level build-covariates /
  conditional-screen commands, all parameterized by run spec.
- backtest.py: keep portfolio accounting shared; only input weights and
  benchmark differ by run.
- tests: add synthetic tests where a raw signal looks predictive until a
  covariate is included, and where a truly incremental signal survives both
  covariate conditioning and novelty checks.

## Definition of Done

- One command can run the same research pipeline for Nasdaq-100 or S&P 500 by
  changing only the run specification.
- Covariate manifests prove every covariate is point-in-time and lagged
  correctly.
- Conditional Fisher simulation holds realized covariates fixed.
- rho_star is reported both before and after accepted-signal conditioning.
- Conditional novelty rejects redundant signals with a recorded nearest match.
- 2026 performance is unavailable to the agent and to signal selection until
  both universe strategies are frozen.
