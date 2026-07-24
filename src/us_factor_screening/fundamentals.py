"""Fundamentals-based factors built from Polygon quarterly financials.

This module adds value, profitability, investment, and accruals factors that
consume point-in-time quarterly financials. Financials are loaded from
per-symbol CSV files (``financials_{SYMBOL}.csv``) produced by the
``download_database.py`` script, then aligned to a daily trading calendar so
that each factor value on date T uses only statements that were publicly
available on or before T.

Quarterly -> daily alignment is done via ``availability_date`` = ``max(updated,
calendarDate + pit_lag_days)``. The ``updated`` field is Polygon's filing
date; when missing we conservatively fall back to ``calendarDate + 45 days``.
For each trading date D the panel surfaces the most recent quarter whose
availability date <= D, guaranteeing point-in-time correctness.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .factor_zoo import (
    FactorContext,
    FactorRegistry,
    _sanitize,
    _spec,
)


@dataclass(frozen=True)
class FundamentalsConfig:
    """Tuning knobs for :class:`FundamentalsPanel`."""

    lookback_quarters: int = 8  # number of trailing quarters retained per symbol
    pit_lag_days: int = 45  # minimum delay between report period end and availability


class FundamentalsPanel:
    """Point-in-time quarterly financials aligned to a daily trading calendar.

    Loads quarterly financials from a directory of per-symbol CSV files (as
    produced by the ``download_database.py`` script), then for each trading
    date provides the most-recently-available financial statement per symbol.
    """

    def __init__(
        self,
        financials_dir: str | Path,
        symbols: list[str],
        config: FundamentalsConfig | None = None,
    ) -> None:
        self.config = config or FundamentalsConfig()
        self.symbols = [str(s).upper() for s in symbols]
        self._raw: dict[str, pd.DataFrame] = {}  # symbol -> quarterly DataFrame
        self._daily: dict[str, pd.DataFrame] = {}  # symbol -> daily forward-filled frame
        for symbol in self.symbols:
            self._load_symbol(symbol, financials_dir)

    # ------------------------------------------------------------------ loading

    def _load_symbol(self, symbol: str, financials_dir: str | Path) -> None:
        """Load one symbol's quarterly financials CSV and compute availability."""
        path = Path(financials_dir) / f"financials_{symbol}.csv"
        if not path.exists():
            self._raw[symbol] = pd.DataFrame()
            return
        df = pd.read_csv(path)
        for col in ("reportPeriod", "calendarDate", "updated", "dateKey"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        sort_col = "reportPeriod" if "reportPeriod" in df.columns else "calendarDate"
        if sort_col in df.columns:
            df = df.sort_values(sort_col).reset_index(drop=True)

        calendar = (
            df["calendarDate"]
            if "calendarDate" in df.columns
            else df.get("reportPeriod")
        )
        if calendar is None:
            self._raw[symbol] = pd.DataFrame()
            return

        min_available = calendar + pd.Timedelta(days=self.config.pit_lag_days)
        if "updated" in df.columns:
            updated_filled = df["updated"].fillna(min_available)
            availability = pd.concat([updated_filled, min_available], axis=1).max(axis=1)
        else:
            availability = min_available
        df = df.assign(_availability=availability)
        df = df.dropna(subset=["_availability"])
        # Keep the latest fiscal period when multiple filings share an availability date.
        if sort_col in df.columns:
            df = df.sort_values(["_availability", sort_col])
        df = df.drop_duplicates(subset=["_availability"], keep="last")
        self._raw[symbol] = df.reset_index(drop=True)

    # ------------------------------------------------------------- core helpers

    def _pit_series(self, symbol: str, field: str, dates: pd.DatetimeIndex) -> pd.Series:
        """Forward-filled point-in-time series for one symbol/field.

        For each date D, returns the value from the most recent quarter whose
        availability date <= D (NaN before the first available quarter).
        """
        raw = self._raw.get(symbol, pd.DataFrame())
        if raw.empty or field not in raw.columns:
            return pd.Series(np.nan, index=dates, name=symbol, dtype=float)
        series = (
            raw.sort_values("_availability")
            .set_index("_availability")[field]
            .astype(float)
        )
        return series.reindex(dates, method="ffill").rename(symbol)

    def align_to_calendar(self, dates: pd.DatetimeIndex) -> None:
        """Forward-fill every quarter's fields onto the daily date index.

        For each date D, the value is from the most recent quarter whose
        availability date <= D. This ensures point-in-time correctness.
        """
        dates = pd.DatetimeIndex(dates)
        for symbol in self.symbols:
            raw = self._raw.get(symbol, pd.DataFrame())
            if raw.empty:
                self._daily[symbol] = pd.DataFrame(index=dates)
                continue
            # Only forward-fill numeric financial fields; skip ticker/period/dates.
            fields = [
                c
                for c in raw.columns
                if c != "_availability" and pd.api.types.is_numeric_dtype(raw[c])
            ]
            frame = pd.DataFrame(
                {field: self._pit_series(symbol, field, dates) for field in fields},
                index=dates,
            )
            self._daily[symbol] = frame

    # ------------------------------------------------------------- public API

    def get_field(self, field: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a Date x Symbol DataFrame for one financial field (latest quarter)."""
        dates = pd.DatetimeIndex(dates)
        series: dict[str, pd.Series] = {}
        for symbol in self.symbols:
            daily = self._daily.get(symbol)
            if (
                daily is not None
                and field in daily.columns
                and daily.index.equals(dates)
            ):
                series[symbol] = daily[field]
            else:
                series[symbol] = self._pit_series(symbol, field, dates)
        return pd.DataFrame(series).reindex(columns=self.symbols)

    def get_trailing_field(
        self,
        field: str,
        dates: pd.DatetimeIndex,
        quarters: int = 4,
    ) -> pd.DataFrame:
        """Return the sum of the last ``quarters`` quarters (e.g. trailing 4Q TTM)."""
        dates = pd.DatetimeIndex(dates)
        result: dict[str, pd.Series] = {}
        for symbol in self.symbols:
            raw = self._raw.get(symbol, pd.DataFrame())
            if raw.empty or field not in raw.columns:
                result[symbol] = pd.Series(np.nan, index=dates, dtype=float)
                continue
            sort_col = "reportPeriod" if "reportPeriod" in raw.columns else "calendarDate"
            df = raw.sort_values(sort_col).copy()
            df["_trailing"] = (
                df[field].astype(float).rolling(quarters, min_periods=quarters).sum()
            )
            df = df.sort_values("_availability")
            indexed = df.set_index("_availability")["_trailing"]
            indexed = indexed[~indexed.index.duplicated(keep="last")]
            result[symbol] = indexed.reindex(dates, method="ffill")
        return pd.DataFrame(result).reindex(columns=self.symbols)

    def get_yoy_growth(
        self,
        field: str,
        dates: pd.DatetimeIndex,
        quarters_back: int = 4,
    ) -> pd.DataFrame:
        """Year-over-year growth: ``(current - N quarters ago) / |N quarters ago|``."""
        dates = pd.DatetimeIndex(dates)
        result: dict[str, pd.Series] = {}
        for symbol in self.symbols:
            raw = self._raw.get(symbol, pd.DataFrame())
            if raw.empty or field not in raw.columns:
                result[symbol] = pd.Series(np.nan, index=dates, dtype=float)
                continue
            sort_col = "reportPeriod" if "reportPeriod" in raw.columns else "calendarDate"
            df = raw.sort_values(sort_col).copy()
            values = df[field].astype(float)
            past = values.shift(quarters_back)
            df["_yoy"] = (values - past) / past.abs().where(past.abs() != 0)
            df = df.sort_values("_availability")
            indexed = df.set_index("_availability")["_yoy"]
            indexed = indexed[~indexed.index.duplicated(keep="last")]
            result[symbol] = indexed.reindex(dates, method="ffill")
        return pd.DataFrame(result).reindex(columns=self.symbols)


# ---------------------------------------------------------------------------
# Factor compute functions
# ---------------------------------------------------------------------------


def _require_fundamentals(context: FactorContext) -> FundamentalsPanel:
    if context.fundamentals is None:
        raise ValueError(
            "Fundamentals factor requires context.fundamentals; pass a "
            "FundamentalsPanel to FactorContext.from_frames (fundamentals=...)."
        )
    return context.fundamentals


def _stock(context: FactorContext, field: str) -> pd.DataFrame:
    """Latest available quarter's value for a stock (balance sheet) field."""
    panel = _require_fundamentals(context)
    return panel.get_field(field, context.close.index).reindex_like(context.close)


def _ttm(context: FactorContext, field: str) -> pd.DataFrame:
    """Trailing twelve-month (sum of last 4 quarters) value for a flow field."""
    panel = _require_fundamentals(context)
    return panel.get_trailing_field(field, context.close.index, 4).reindex_like(
        context.close
    )


def _yoy(context: FactorContext, field: str) -> pd.DataFrame:
    """Year-over-year growth (current vs. 4 quarters ago) for a field."""
    panel = _require_fundamentals(context)
    return panel.get_yoy_growth(field, context.close.index, 4).reindex_like(
        context.close
    )


def _market_cap(context: FactorContext) -> pd.DataFrame:
    shares = _stock(context, "weightedAverageSharesOutstanding")
    return shares * context.close


# --- VALUE family ---------------------------------------------------------


def _book_to_market(context: FactorContext) -> pd.DataFrame:
    book = _stock(context, "bookValue")
    mcap = _market_cap(context)
    return _sanitize(book / mcap.where(mcap > 0))


def _earnings_yield(context: FactorContext) -> pd.DataFrame:
    eps_ttm = _ttm(context, "eps")
    return _sanitize(eps_ttm / context.close.where(context.close > 0))


def _cash_flow_yield(context: FactorContext) -> pd.DataFrame:
    ocf_ttm = _ttm(context, "netCashFromOperatingActivities")
    mcap = _market_cap(context)
    return _sanitize(ocf_ttm / mcap.where(mcap > 0))


def _sales_to_price(context: FactorContext) -> pd.DataFrame:
    rev_ttm = _ttm(context, "revenue")
    mcap = _market_cap(context)
    return _sanitize(rev_ttm / mcap.where(mcap > 0))


def _dividend_yield(context: FactorContext) -> pd.DataFrame:
    dps_ttm = _ttm(context, "dividendPerShare")
    return _sanitize(dps_ttm / context.close.where(context.close > 0))


# --- PROFITABILITY family -------------------------------------------------


def _roe(context: FactorContext) -> pd.DataFrame:
    ni_ttm = _ttm(context, "netIncome")
    book = _stock(context, "bookValue")
    return _sanitize(ni_ttm / book.where(book != 0))


def _roa(context: FactorContext) -> pd.DataFrame:
    ni_ttm = _ttm(context, "netIncome")
    assets = _stock(context, "assets")
    return _sanitize(ni_ttm / assets.where(assets != 0))


def _gross_profitability(context: FactorContext) -> pd.DataFrame:
    gp_ttm = _ttm(context, "grossProfit")
    assets = _stock(context, "assets")
    return _sanitize(gp_ttm / assets.where(assets != 0))


def _operating_margin(context: FactorContext) -> pd.DataFrame:
    op_ttm = _ttm(context, "operatingIncome")
    rev_ttm = _ttm(context, "revenue")
    return _sanitize(op_ttm / rev_ttm.where(rev_ttm != 0))


def _net_margin(context: FactorContext) -> pd.DataFrame:
    ni_ttm = _ttm(context, "netIncome")
    rev_ttm = _ttm(context, "revenue")
    return _sanitize(ni_ttm / rev_ttm.where(rev_ttm != 0))


# --- INVESTMENT family ----------------------------------------------------


def _asset_growth(context: FactorContext) -> pd.DataFrame:
    return _sanitize(_yoy(context, "assets"))


def _equity_issuance(context: FactorContext) -> pd.DataFrame:
    return _sanitize(_yoy(context, "weightedAverageSharesOutstanding"))


def _investment_to_assets(context: FactorContext) -> pd.DataFrame:
    capex_ttm = _ttm(context, "capitalExpenditure")
    assets = _stock(context, "assets")
    return _sanitize(capex_ttm / assets.where(assets != 0))


# --- ACCRUALS family ------------------------------------------------------


def _sloan_accruals(context: FactorContext) -> pd.DataFrame:
    ni_ttm = _ttm(context, "netIncome")
    ocf_ttm = _ttm(context, "netCashFromOperatingActivities")
    assets = _stock(context, "assets")
    return _sanitize((ni_ttm - ocf_ttm) / assets.where(assets != 0))


def _cash_based_operating_profitability(context: FactorContext) -> pd.DataFrame:
    ocf_ttm = _ttm(context, "netCashFromOperatingActivities")
    assets = _stock(context, "assets")
    return _sanitize(ocf_ttm / assets.where(assets != 0))


FUNDAMENTALS_REGISTRY = FactorRegistry(
    [
        _spec(
            "book_to_market",
            "value",
            "Book value to market capitalization",
            "bookValue/(shares*C)",
            63,
            ("Close",),
            1,
            _book_to_market,
        ),
        _spec(
            "earnings_yield",
            "value",
            "Trailing EPS divided by close price",
            "EPS_TTM/C",
            252,
            ("Close",),
            1,
            _earnings_yield,
        ),
        _spec(
            "cash_flow_yield",
            "value",
            "Trailing operating cash flow to market cap",
            "OCF_TTM/MCAP",
            252,
            ("Close",),
            1,
            _cash_flow_yield,
        ),
        _spec(
            "sales_to_price",
            "value",
            "Trailing revenue to market cap",
            "REV_TTM/MCAP",
            252,
            ("Close",),
            1,
            _sales_to_price,
        ),
        _spec(
            "dividend_yield",
            "value",
            "Trailing dividend per share to close price",
            "DPS_TTM/C",
            252,
            ("Close",),
            1,
            _dividend_yield,
        ),
        _spec(
            "roe",
            "profitability",
            "Trailing net income to book value",
            "NI_TTM/bookValue",
            252,
            ("Close",),
            1,
            _roe,
        ),
        _spec(
            "roa",
            "profitability",
            "Trailing net income to assets",
            "NI_TTM/assets",
            252,
            ("Close",),
            1,
            _roa,
        ),
        _spec(
            "gross_profitability",
            "profitability",
            "Trailing gross profit to assets (Novy-Marx 2013)",
            "GP_TTM/assets",
            252,
            ("Close",),
            1,
            _gross_profitability,
        ),
        _spec(
            "operating_margin",
            "profitability",
            "Trailing operating income to revenue",
            "OI_TTM/REV_TTM",
            252,
            ("Close",),
            1,
            _operating_margin,
        ),
        _spec(
            "net_margin",
            "profitability",
            "Trailing net income to revenue",
            "NI_TTM/REV_TTM",
            252,
            ("Close",),
            1,
            _net_margin,
        ),
        _spec(
            "asset_growth",
            "investment",
            "Year-over-year asset growth (Cooper, Gulen & Schill 2008)",
            "(assets_t-assets_t-4)/|assets_t-4|",
            504,
            ("Close",),
            -1,
            _asset_growth,
        ),
        _spec(
            "equity_issuance",
            "investment",
            "Year-over-year share count growth",
            "(shares_t-shares_t-4)/|shares_t-4|",
            504,
            ("Close",),
            -1,
            _equity_issuance,
        ),
        _spec(
            "investment_to_assets",
            "investment",
            "Trailing capex to assets (Lyandres, Sun & Zhang 2008)",
            "CAPEX_TTM/assets",
            252,
            ("Close",),
            -1,
            _investment_to_assets,
        ),
        _spec(
            "sloan_accruals",
            "accruals",
            "Sloan accruals: (NI-OCF)/assets (Sloan 1996)",
            "(NI_TTM-OCF_TTM)/assets",
            252,
            ("Close",),
            -1,
            _sloan_accruals,
        ),
        _spec(
            "cash_based_operating_profitability",
            "accruals",
            "Cash-based operating profitability (Ball et al. 2016)",
            "OCF_TTM/assets",
            252,
            ("Close",),
            1,
            _cash_based_operating_profitability,
        ),
    ]
)


def compute_fundamental_factors(
    frames: dict[str, pd.DataFrame],
    fundamentals: FundamentalsPanel,
    names: Iterable[str] | None = None,
    *,
    benchmark: str = "SPY",
    oriented: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute selected fundamentals-registered factors as Date x Symbol matrices."""
    selected = list(names) if names is not None else FUNDAMENTALS_REGISTRY.names()
    context = FactorContext.from_frames(
        frames, benchmark=benchmark, fundamentals=fundamentals
    )
    output: dict[str, pd.DataFrame] = {}
    for name in selected:
        spec = FUNDAMENTALS_REGISTRY.get(name)
        values = _sanitize(spec.compute(context)).reindex_like(context.close)
        if oriented and spec.direction == -1:
            values = -values
        output[name] = values
    return output


def orient_fundamental_factor(values: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """Orient a fundamentals factor so larger values follow its conventional long side."""
    spec = FUNDAMENTALS_REGISTRY.get(factor_name)
    return -values if spec.direction == -1 else values.copy()
