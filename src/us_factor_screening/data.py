from __future__ import annotations

import os
import random
import re
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
_US_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")
_NON_US_SUFFIXES = (
    ".AX",
    ".BO",
    ".HK",
    ".L",
    ".NS",
    ".SS",
    ".SZ",
    ".T",
    ".TO",
)
MAX_STALE_CALENDAR_DAYS = 10


class MarketDataError(RuntimeError):
    """Raised when a source cannot provide usable market data."""


class MarketDataRateLimitError(MarketDataError):
    """Raised when a provider rejects a request because of throttling."""


@dataclass(frozen=True)
class MarketDataProvenance:
    """The data definition that must remain uniform across a research panel."""

    provider: str
    feed: str
    adjustment: str
    interval: str = "1d"
    session: str = "regular"

    def __post_init__(self) -> None:
        for name in ("provider", "feed", "adjustment", "interval", "session"):
            value = str(getattr(self, name)).strip().lower()
            if not value:
                raise ValueError(f"market-data {name} must not be empty")
            object.__setattr__(self, name, value)

    def as_record(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "feed": self.feed,
            "adjustment": self.adjustment,
            "interval": self.interval,
            "session": self.session,
        }

    @classmethod
    def from_record(cls, values: Mapping[str, Any]) -> MarketDataProvenance:
        try:
            return cls(
                provider=str(values["provider"]),
                feed=str(values["feed"]),
                adjustment=str(values["adjustment"]),
                interval=str(values.get("interval", "1d")),
                session=str(values.get("session", "regular")),
            )
        except KeyError as exc:
            raise ValueError(f"market-data provenance is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class MarketDataResult:
    """One normalized symbol response with source identity and retrieval metadata."""

    symbol: str
    frame: pd.DataFrame
    provenance: MarketDataProvenance
    retrieved_at: datetime
    cache_hit: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbol = validate_us_symbol(self.symbol)
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("market-data retrieved_at must be timezone-aware")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "retrieved_at", self.retrieved_at.astimezone(UTC))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class MarketDataPanel(Mapping[str, pd.DataFrame]):
    """Mapping-compatible panel that rejects mixed provider definitions."""

    results: Mapping[str, MarketDataResult]

    def __post_init__(self) -> None:
        normalized: dict[str, MarketDataResult] = {}
        for key, result in self.results.items():
            symbol = validate_us_symbol(key)
            if symbol in normalized:
                raise ValueError(f"duplicate market-data result for {symbol}")
            if result.symbol != symbol:
                raise ValueError(
                    f"market-data key {symbol!r} does not match result symbol {result.symbol!r}"
                )
            normalized[symbol] = result
        if not normalized:
            raise ValueError("market-data panel must contain at least one symbol")
        identities = {result.provenance for result in normalized.values()}
        if len(identities) != 1:
            descriptions = sorted(str(identity.as_record()) for identity in identities)
            raise MarketDataError(
                "mixed market-data definitions are not allowed: " + "; ".join(descriptions)
            )
        object.__setattr__(self, "results", MappingProxyType(normalized))

    def __getitem__(self, symbol: str) -> pd.DataFrame:
        return self.results[symbol].frame

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def provenance(self) -> MarketDataProvenance:
        return next(iter(self.results.values())).provenance

    @classmethod
    def from_results(cls, results: Iterable[MarketDataResult]) -> MarketDataPanel:
        materialized = list(results)
        symbols = [result.symbol for result in materialized]
        if len(symbols) != len(set(symbols)):
            raise ValueError("duplicate symbol results are not allowed")
        return cls({result.symbol: result for result in materialized})

    def manifest(self) -> dict[str, Any]:
        symbols: dict[str, dict[str, Any]] = {}
        for symbol, result in self.results.items():
            frame = result.frame
            symbols[symbol] = {
                "rows": len(frame),
                "first_date": frame["Date"].min().strftime("%Y-%m-%d"),
                "last_date": frame["Date"].max().strftime("%Y-%m-%d"),
                "retrieved_at": result.retrieved_at.isoformat(),
                "cache_hit": result.cache_hit,
                "metadata": dict(result.metadata),
            }
        return {
            "schema_version": 1,
            "provenance": self.provenance.as_record(),
            "symbols": symbols,
        }


class MarketDataSource(ABC):
    """Common contract for one-definition daily US market-data sources."""

    @property
    @abstractmethod
    def provenance(self) -> MarketDataProvenance:
        """Return the provider definition used for every fetched symbol."""

    @abstractmethod
    def fetch_result(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        refresh: bool = False,
    ) -> MarketDataResult:
        """Fetch one symbol with provenance and retrieval metadata."""

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        return self.fetch_result(
            symbol,
            start_date,
            end_date,
            refresh=refresh,
        ).frame

    def fetch_many(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        refresh: bool = False,
    ) -> MarketDataPanel:
        normalized = list(dict.fromkeys(validate_us_symbol(symbol) for symbol in symbols))
        if not normalized:
            raise ValueError("At least one symbol is required")
        return MarketDataPanel.from_results(
            self.fetch_result(symbol, start_date, end_date, refresh=refresh)
            for symbol in normalized
        )


@dataclass(frozen=True)
class DataQualityReport:
    symbol: str
    rows: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    duplicate_dates: int
    invalid_ohlc_rows: int
    nonpositive_close_rows: int
    negative_volume_rows: int
    missing_value_rows: int

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.rows == 0,
                self.duplicate_dates,
                self.invalid_ohlc_rows,
                self.nonpositive_close_rows,
                self.negative_volume_rows,
                self.missing_value_rows,
            )
        )

    def as_record(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "ok": self.ok,
            "rows": self.rows,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "duplicate_dates": self.duplicate_dates,
            "invalid_ohlc_rows": self.invalid_ohlc_rows,
            "nonpositive_close_rows": self.nonpositive_close_rows,
            "negative_volume_rows": self.negative_volume_rows,
            "missing_value_rows": self.missing_value_rows,
        }


def validate_us_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if normalized.endswith(_NON_US_SUFFIXES):
        raise ValueError(f"Non-US exchange suffix is outside this project's scope: {symbol!r}")
    if not _US_SYMBOL.fullmatch(normalized):
        raise ValueError(f"Unsupported US-listed symbol: {symbol!r}")
    return normalized


def normalize_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize one ticker to a deterministic Date + OHLCV schema."""
    if frame is None or frame.empty:
        raise MarketDataError(f"{symbol}: market-data response is empty")

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "Date" not in data.columns:
        data = data.reset_index()
        if "Date" not in data.columns:
            data = data.rename(columns={data.columns[0]: "Date"})

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise MarketDataError(f"{symbol}: missing required columns: {', '.join(missing)}")

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    numeric = [column for column in REQUIRED_COLUMNS if column != "Date"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if data.empty:
        raise MarketDataError(f"{symbol}: no rows have a valid date")
    return data


def validate_ohlcv(frame: pd.DataFrame, symbol: str) -> DataQualityReport:
    """Return structural and market-invariant checks for normalized OHLCV."""
    data = frame.copy()
    missing_values = int(data[list(REQUIRED_COLUMNS)].isna().any(axis=1).sum())
    high_floor = data[["Open", "Low", "Close"]].max(axis=1)
    low_ceiling = data[["Open", "High", "Close"]].min(axis=1)
    invalid_ohlc = int(((data["High"] < high_floor) | (data["Low"] > low_ceiling)).sum())
    duplicate_dates = int(data["Date"].duplicated().sum())

    return DataQualityReport(
        symbol=symbol,
        rows=len(data),
        first_date=data["Date"].min() if len(data) else None,
        last_date=data["Date"].max() if len(data) else None,
        duplicate_dates=duplicate_dates,
        invalid_ohlc_rows=invalid_ohlc,
        nonpositive_close_rows=int((data["Close"] <= 0).sum()),
        negative_volume_rows=int((data["Volume"] < 0).sum()),
        missing_value_rows=missing_values,
    )


def assert_not_stale(
    data: pd.DataFrame,
    symbol: str,
    requested_end: pd.Timestamp,
    *,
    provider: str,
) -> None:
    latest = data["Date"].max().normalize()
    effective_end = min(requested_end.normalize(), pd.Timestamp.today().normalize())
    stale_days = (effective_end - latest).days
    if stale_days > MAX_STALE_CALENDAR_DAYS:
        raise MarketDataError(
            f"{symbol}: latest {provider} row is {latest.date()}, {stale_days} calendar days "
            f"before requested end {effective_end.date()} (stale)"
        )


class YahooFinanceSource(MarketDataSource):
    """Standalone, cached Yahoo Finance OHLCV provider for this project."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/market_data",
        fetcher: Callable[[str, str, str], pd.DataFrame] | None = None,
        *,
        max_retries: int = 4,
        retry_base_delay: float = 30.0,
        retry_max_delay: float = 600.0,
        inter_request_delay: float = 1.5,
        jitter: float = 0.25,
        retry_after_resolver: Callable[[BaseException], float | None] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._fetcher = fetcher
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay cannot be negative")
        if retry_max_delay < 0:
            raise ValueError("retry_max_delay cannot be negative")
        if inter_request_delay < 0:
            raise ValueError("inter_request_delay cannot be negative")
        if not 0.0 <= jitter <= 1.0:
            raise ValueError("jitter must be between 0.0 and 1.0")
        self.max_retries = max_retries
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.inter_request_delay = float(inter_request_delay)
        self.jitter = float(jitter)
        self._retry_after_resolver = retry_after_resolver
        self._sleeper = sleeper
        # yfinance's YFRateLimitError does not surface the underlying HTTP
        # response, so the Retry-After header cannot be read directly. The
        # resolver hook lets callers that wrap yfinance's session feed back an
        # explicit retry-after value in seconds.
        self._rng = rng if rng is not None else random.random

    @property
    def provenance(self) -> MarketDataProvenance:
        return MarketDataProvenance(
            provider="yahoo_finance",
            feed="yahoo_chart",
            adjustment="backward_total_return",
        )

    def _cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        key = f"{symbol}_{start_date}_{end_date}".replace("-", "")
        return self.cache_dir / f"{key}.csv"

    def _apply_jitter(self, delay: float) -> float:
        if self.jitter <= 0.0:
            return max(0.0, delay)
        # _rng returns a float in [0, 1); remap to [-jitter, +jitter] and scale.
        offset = (self._rng() * 2.0 - 1.0) * self.jitter
        return max(0.0, delay * (1.0 + offset))

    def _compute_rate_limit_delay(self, attempt: int, exc: BaseException) -> float:
        """Pick the next 429 backoff: explicit Retry-After wins, else exponential."""
        if self._retry_after_resolver is not None:
            try:
                retry_after = self._retry_after_resolver(exc)
            except Exception:  # pragma: no cover - defensive: resolver bugs must not crash the loop
                retry_after = None
            if retry_after is not None and retry_after > 0:
                return float(retry_after)
        base = self.retry_base_delay * (2.0 ** attempt)
        capped = min(base, self.retry_max_delay)
        return self._apply_jitter(capped)

    def _download(self, symbol: str, start_date: str, end_exclusive: str) -> pd.DataFrame:
        if self._fetcher is not None:
            return self._fetcher(symbol, start_date, end_exclusive)
        # Yahoo's adjusted-close ratio applies splits and cash distributions
        # backward while keeping the latest close on its current share basis.
        return yf.Ticker(symbol).history(
            start=start_date,
            end=end_exclusive,
            auto_adjust=True,
            actions=False,
        )

    def _download_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_exclusive: str,
    ) -> pd.DataFrame:
        for attempt in range(self.max_retries + 1):
            try:
                frame = self._download(symbol, start_date, end_exclusive)
            except YFRateLimitError as exc:
                if attempt >= self.max_retries:
                    raise
                # Yahoo's 429 needs much longer backoff than a generic 2s retry:
                # 30s -> 60s -> 120s -> 240s (capped at retry_max_delay), with an
                # optional Retry-After override and ±jitter% spread to avoid
                # thundering-herd retries across symbols.
                self._sleeper(self._compute_rate_limit_delay(attempt, exc))
                continue
            # Serial throttling between successful symbol fetches keeps the
            # sustained request rate below Yahoo's per-IP 429 trigger.
            if self.inter_request_delay > 0.0:
                self._sleeper(self._apply_jitter(self.inter_request_delay))
            return frame
        raise AssertionError("unreachable")

    def fetch_result(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        refresh: bool = False,
    ) -> MarketDataResult:
        symbol = validate_us_symbol(symbol)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start > end:
            raise ValueError("start_date must be on or before end_date")
        start_text = start.strftime("%Y-%m-%d")
        end_text = end.strftime("%Y-%m-%d")
        cache_path = self._cache_path(symbol, start_text, end_text)

        if cache_path.exists() and not refresh:
            raw_frame = pd.read_csv(cache_path)
            cache_hit = True
            retrieved_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)
        else:
            end_exclusive = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                raw_frame = self._download_with_retry(symbol, start_text, end_exclusive)
            except Exception as exc:
                raise MarketDataError(f"{symbol}: Yahoo Finance request failed: {exc}") from exc
            if raw_frame is None or raw_frame.empty:
                raise MarketDataError(f"{symbol}: Yahoo Finance returned no rows")
            cache_hit = False
            retrieved_at = datetime.now(UTC)

        data = normalize_ohlcv(raw_frame, symbol)
        data = data[(data["Date"] >= start) & (data["Date"] <= end)].reset_index(drop=True)
        if data.empty:
            raise MarketDataError(f"{symbol}: no rows in requested date range")
        assert_not_stale(data, symbol, end, provider="Yahoo")

        report = validate_ohlcv(data, symbol)
        if not report.ok:
            raise MarketDataError(f"{symbol}: OHLCV validation failed: {report.as_record()}")

        if not cache_path.exists() or refresh:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _write_csv_atomic(data, cache_path)
        return MarketDataResult(
            symbol=symbol,
            frame=data,
            provenance=self.provenance,
            retrieved_at=retrieved_at,
            cache_hit=cache_hit,
            metadata={
                "cache_path": str(cache_path),
                "price_basis": "backward_total_return",
                "corporate_actions": "splits_and_cash_distributions",
            },
        )


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        compression = "gzip" if path.suffix == ".gz" else None
        frame.to_csv(temporary_name, index=False, compression=compression)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def close_matrix(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Align normalized close prices by trading date without backward filling."""
    if not frames:
        raise ValueError("At least one symbol is required")
    series = []
    for symbol, frame in frames.items():
        normalized = normalize_ohlcv(frame, symbol)
        values = normalized.set_index("Date")["Close"].rename(symbol)
        series.append(values)
    return pd.concat(series, axis=1).sort_index().ffill(limit=3)
