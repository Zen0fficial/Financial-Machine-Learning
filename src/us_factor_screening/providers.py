from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import (
    MarketDataError,
    MarketDataProvenance,
    MarketDataRateLimitError,
    MarketDataResult,
    MarketDataSource,
    _write_csv_atomic,
    assert_not_stale,
    normalize_ohlcv,
    validate_ohlcv,
    validate_us_symbol,
)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


HttpTransport = Callable[[Request, float], HttpResponse]


def urllib_transport(request: Request, timeout: float) -> HttpResponse:
    try:
        with urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=int(response.status),
                body=response.read(),
                headers=dict(response.headers.items()),
            )
    except HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            body=exc.read(),
            headers=dict(exc.headers.items()) if exc.headers else {},
        )
    except (URLError, OSError) as exc:
        raise MarketDataError(f"market-data request failed: {exc}") from exc


class AlpacaMarketDataSource(MarketDataSource):
    """Daily US stock bars from one explicit Alpaca feed and adjustment mode."""

    _FEEDS = frozenset({"iex", "sip"})
    _ADJUSTMENTS = frozenset({"raw", "split", "dividend", "all"})

    def __init__(
        self,
        cache_dir: str | Path = ".cache/market_data",
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        feed: str = "iex",
        adjustment: str = "all",
        endpoint: str = "https://data.alpaca.markets/v2/stocks",
        timeout: float = 20.0,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        transport: HttpTransport = urllib_transport,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        normalized_feed = str(feed).strip().lower()
        normalized_adjustment = str(adjustment).strip().lower()
        if normalized_feed not in self._FEEDS:
            raise ValueError(f"Alpaca feed must be one of {sorted(self._FEEDS)}")
        if normalized_adjustment not in self._ADJUSTMENTS:
            raise ValueError(
                f"Alpaca adjustment must be one of {sorted(self._ADJUSTMENTS)}"
            )
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay cannot be negative")
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key or os.getenv("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY")
        self.feed = normalized_feed
        self.adjustment = normalized_adjustment
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.transport = transport
        self.sleeper = sleeper
        self.clock = clock or (lambda: datetime.now(UTC))

    @property
    def provenance(self) -> MarketDataProvenance:
        return MarketDataProvenance(
            provider="alpaca",
            feed=self.feed,
            adjustment=self.adjustment,
        )

    def _cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        key = (
            f"alpaca_{self.feed}_{self.adjustment}_{symbol}_{start_date}_{end_date}"
        ).replace("-", "")
        return self.cache_dir / f"{key}.csv"

    def _decode_response(self, response: HttpResponse) -> dict[str, Any]:
        if response.status == 429:
            raise MarketDataRateLimitError("Alpaca rate limit reached")
        if response.status in {401, 403}:
            raise MarketDataError(
                "Alpaca authentication or feed entitlement failed; check API credentials "
                f"and access to feed {self.feed!r}"
            )
        if response.status >= 400:
            detail = response.body.decode("utf-8", errors="replace")[:300].strip()
            raise MarketDataError(
                f"Alpaca returned HTTP {response.status}: {detail or 'empty response'}"
            )
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MarketDataError("Alpaca returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise MarketDataError("Alpaca returned a non-object JSON payload")
        return payload

    def _download(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        if not self.api_key or not self.secret_key:
            raise MarketDataError(
                "Alpaca requires APCA_API_KEY_ID and APCA_API_SECRET_KEY; cached files "
                "can still be replayed without credentials"
            )
        start = pd.Timestamp(start_date)
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        params: dict[str, str | int] = {
            "timeframe": "1Day",
            "start": start.tz_localize("UTC").isoformat(),
            "end": end_exclusive.tz_localize("UTC").isoformat(),
            "limit": 10000,
            "adjustment": self.adjustment,
            "feed": self.feed,
            "sort": "asc",
        }
        headers = {
            "Accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "User-Agent": "us-factor-screening/0.1",
        }
        raw_bars: list[dict[str, Any]] = []
        request_ids: list[str] = []
        page_token: str | None = None
        for _ in range(100):
            page_params = dict(params)
            if page_token:
                page_params["page_token"] = page_token
            url = f"{self.endpoint}/{quote(symbol, safe='')}/bars?{urlencode(page_params)}"
            response = self.transport(Request(url, headers=headers), self.timeout)
            request_id = response.headers.get("X-Request-ID") or response.headers.get(
                "x-request-id"
            )
            if request_id:
                request_ids.append(str(request_id))
            payload = self._decode_response(response)
            page = payload.get("bars")
            if not isinstance(page, list):
                raise MarketDataError("Alpaca response omitted a valid bars list")
            raw_bars.extend(item for item in page if isinstance(item, dict))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        else:
            raise MarketDataError("Alpaca pagination exceeded 100 pages")
        if not raw_bars:
            raise MarketDataError(f"{symbol}: Alpaca returned no bars")
        frame = pd.DataFrame.from_records(raw_bars).rename(
            columns={
                "t": "Date",
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
            }
        )
        return frame, request_ids

    def _download_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        for attempt in range(self.max_retries + 1):
            try:
                return self._download(symbol, start_date, end_date)
            except MarketDataRateLimitError:
                if attempt >= self.max_retries:
                    raise
                self.sleeper(self.retry_base_delay * (2**attempt))
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
        request_ids: list[str] = []
        if cache_path.exists() and not refresh:
            raw_frame = pd.read_csv(cache_path)
            cache_hit = True
            retrieved_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)
        else:
            raw_frame, request_ids = self._download_with_retry(symbol, start_text, end_text)
            cache_hit = False
            retrieved_at = self.clock().astimezone(UTC)

        data = normalize_ohlcv(raw_frame, symbol)
        data = data[(data["Date"] >= start) & (data["Date"] <= end)].reset_index(drop=True)
        if data.empty:
            raise MarketDataError(f"{symbol}: no Alpaca rows in requested date range")
        assert_not_stale(data, symbol, end, provider="Alpaca")
        report = validate_ohlcv(data, symbol)
        if not report.ok:
            raise MarketDataError(f"{symbol}: Alpaca OHLCV validation failed: {report.as_record()}")
        if not cache_path.exists() or refresh:
            _write_csv_atomic(data, cache_path)
        return MarketDataResult(
            symbol=symbol,
            frame=data,
            provenance=self.provenance,
            retrieved_at=retrieved_at,
            cache_hit=cache_hit,
            metadata={
                "cache_path": str(cache_path),
                "endpoint": self.endpoint,
                "request_ids": request_ids,
            },
        )


class PolygonMarketDataSource(MarketDataSource):
    """Daily or intraday US stock bars from the Polygon shared API."""

    _INTERVALS = {
        "1d": "1/day",
        "1m": "1/minute",
        "5m": "5/minute",
        "15m": "15/minute",
        "30m": "30/minute",
    }

    def __init__(
        self,
        cache_dir: str | Path = ".cache/market_data",
        api_key: str | None = None,
        *,
        base_url: str = "https://api.massiveprivateserver.site",
        interval: str = "1d",
        adjusted: bool = True,
        timeout: float = 30.0,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        inter_request_delay: float = 0.2,
        transport: HttpTransport = urllib_transport,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        normalized_interval = str(interval).strip().lower()
        if normalized_interval not in self._INTERVALS:
            raise ValueError(
                f"Polygon interval must be one of {sorted(self._INTERVALS)}"
            )
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay cannot be negative")
        if retry_max_delay < 0:
            raise ValueError("retry_max_delay cannot be negative")
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise MarketDataError(
                "Polygon requires an api_key argument or POLYGON_API_KEY env var"
            )
        self.base_url = base_url.rstrip("/")
        self.interval = normalized_interval
        self.adjusted = bool(adjusted)
        self.adjustment = "backward_total_return" if self.adjusted else "raw"
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.inter_request_delay = float(inter_request_delay)
        self.transport = transport
        self.sleeper = sleeper
        self.clock = clock or (lambda: datetime.now(UTC))

    @property
    def provenance(self) -> MarketDataProvenance:
        return MarketDataProvenance(
            provider="polygon",
            feed="aggregates",
            adjustment=self.adjustment,
            interval=self.interval,
            session="regular",
        )

    def _cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        key = (
            f"polygon_{self.interval}_{self.adjustment}_{symbol}_{start_date}_{end_date}"
        ).replace("-", "")
        suffix = ".csv.gz" if self.interval != "1d" else ".csv"
        return self.cache_dir / f"{key}{suffix}"

    def _append_api_key(self, url: str) -> str:
        if "apiKey=" in url:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}apiKey={self.api_key}"

    def _build_url(self, symbol: str, start_date: str, end_date: str) -> str:
        range_path = self._INTERVALS[self.interval]
        adjusted_str = "true" if self.adjusted else "false"
        return (
            f"{self.base_url}/v2/aggs/ticker/{quote(symbol, safe='')}/range/"
            f"{range_path}/{start_date}/{end_date}"
            f"?adjusted={adjusted_str}&sort=asc&limit=50000&apiKey={self.api_key}"
        )

    def _parse_retry_after(self, headers: Mapping[str, str]) -> float | None:
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _request_with_retry(self, url: str) -> HttpResponse:
        for attempt in range(self.max_retries + 1):
            response = self.transport(
                Request(url, headers={"User-Agent": "us-factor-screening/0.1"}),
                self.timeout,
            )
            if response.status != 429:
                return response
            if attempt >= self.max_retries:
                raise MarketDataRateLimitError(
                    "Polygon rate limit reached after retries"
                )
            delay = self._parse_retry_after(response.headers)
            if delay is None:
                delay = min(
                    self.retry_base_delay * (2.0 ** attempt),
                    self.retry_max_delay,
                )
            self.sleeper(delay)
        raise AssertionError("unreachable")

    def _decode_response(self, response: HttpResponse) -> dict[str, Any]:
        if response.status in {401, 403}:
            raise MarketDataError(
                "Polygon authentication failed; check the api_key or POLYGON_API_KEY env var"
            )
        if response.status >= 400:
            detail = response.body.decode("utf-8", errors="replace")[:300].strip()
            raise MarketDataError(
                f"Polygon returned HTTP {response.status}: {detail or 'empty response'}"
            )
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MarketDataError("Polygon returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise MarketDataError("Polygon returned a non-object JSON payload")
        return payload

    def _fetch_paginated(self, url: str) -> list[dict[str, Any]]:
        raw_bars: list[dict[str, Any]] = []
        next_url: str | None = url
        for _ in range(100):
            next_url = self._append_api_key(next_url)
            response = self._request_with_retry(next_url)
            payload = self._decode_response(response)
            results = payload.get("results")
            if not isinstance(results, list):
                raise MarketDataError("Polygon response omitted a valid results list")
            raw_bars.extend(item for item in results if isinstance(item, dict))
            next_url = payload.get("next_url")
            if not next_url:
                break
            # Polygon's next_url points to the real api.polygon.io host,
            # not this shared proxy. Rewrite the host to keep using the proxy.
            next_url = self._rewrite_next_url(next_url)
            if self.inter_request_delay > 0.0:
                self.sleeper(self.inter_request_delay)
        else:
            raise MarketDataError("Polygon pagination exceeded 100 pages")
        return raw_bars

    def _rewrite_next_url(self, next_url: str) -> str:
        """Replace the upstream Polygon host with this proxy's base_url."""
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(next_url)
        base_parts = urlsplit(self.base_url)
        # Preserve the path + query from next_url but use our proxy's scheme/netloc.
        return urlunsplit(
            (base_parts.scheme, base_parts.netloc, parts.path, parts.query, parts.fragment)
        )

    def _parse_results(self, raw_bars: list[dict[str, Any]]) -> pd.DataFrame:
        if not raw_bars:
            raise MarketDataError("Polygon returned no bars")
        frame = pd.DataFrame.from_records(raw_bars)
        if "t" not in frame.columns:
            raise MarketDataError("Polygon response omitted the timestamp field 't'")
        frame["Date"] = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_localize(None)
        if self.interval == "1d":
            frame["Date"] = frame["Date"].dt.normalize()
        frame = frame.rename(
            columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
                "vw": "VWAP",
                "n": "TradeCount",
            }
        )
        return frame

    def _download(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.api_key:
            raise MarketDataError(
                "Polygon requires an api_key argument or POLYGON_API_KEY env var"
            )
        url = self._build_url(symbol, start_date, end_date)
        raw_bars = self._fetch_paginated(url)
        return self._parse_results(raw_bars)

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
            if "Date" in raw_frame.columns:
                raw_frame["Date"] = pd.to_datetime(raw_frame["Date"])
            cache_hit = True
            retrieved_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)
        else:
            raw_frame = self._download(symbol, start_text, end_text)
            if raw_frame is None or raw_frame.empty:
                raise MarketDataError(f"{symbol}: Polygon returned no rows")
            cache_hit = False
            retrieved_at = self.clock().astimezone(UTC)

        # Preserve Polygon-specific extra columns (VWAP, TradeCount) across
        # normalization; normalize_ohlcv keeps only the required OHLCV columns.
        extra_cols = [c for c in ("VWAP", "TradeCount") if c in raw_frame.columns]
        extras = raw_frame[["Date", *extra_cols]].copy() if extra_cols else None

        data = normalize_ohlcv(raw_frame, symbol)
        data = data[(data["Date"] >= start) & (data["Date"] <= end)].reset_index(drop=True)
        if data.empty:
            raise MarketDataError(f"{symbol}: no Polygon rows in requested date range")
        assert_not_stale(data, symbol, end, provider="Polygon")

        if extras is not None and extra_cols:
            extras["Date"] = pd.to_datetime(extras["Date"])
            if self.interval == "1d":
                extras["Date"] = extras["Date"].dt.normalize()
            data = data.merge(extras, on="Date", how="left")

        report = validate_ohlcv(data, symbol)
        if not report.ok:
            raise MarketDataError(
                f"{symbol}: Polygon OHLCV validation failed: {report.as_record()}"
            )
        if not cache_path.exists() or refresh:
            _write_csv_atomic(data, cache_path)
        return MarketDataResult(
            symbol=symbol,
            frame=data,
            provenance=self.provenance,
            retrieved_at=retrieved_at,
            cache_hit=cache_hit,
            metadata={
                "cache_path": str(cache_path),
                "endpoint": self.base_url,
                "interval": self.interval,
                "adjustment": self.adjustment,
            },
        )

    # -- Polygon-specific option extensions (not part of MarketDataSource ABC) --

    def fetch_option_contracts(
        self,
        underlying: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch all option contracts for an underlying ticker."""
        underlying = validate_us_symbol(underlying)
        cache_path = self.cache_dir / f"options_contracts_{underlying}.csv"
        if cache_path.exists() and not refresh:
            return pd.read_csv(cache_path)
        url = (
            f"{self.base_url}/v3/reference/options/contracts"
            f"?underlying_ticker={quote(underlying, safe='')}"
            f"&limit=1000&apiKey={self.api_key}"
        )
        raw = self._fetch_paginated(url)
        frame = pd.DataFrame.from_records(raw)
        columns = [
            "ticker",
            "contract_type",
            "exercise_style",
            "expiration_date",
            "strike_price",
            "shares_per_contract",
            "primary_exchange",
            "underlying_ticker",
        ]
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame.loc[:, columns]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _write_csv_atomic(frame, cache_path)
        return frame

    def fetch_option_aggregates(
        self,
        contract_ticker: str,
        start_date: str,
        end_date: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for an option contract."""
        cache_key = contract_ticker.replace(":", "_")
        cache_path = (
            self.cache_dir
            / f"polygon_option_{cache_key}_{start_date}_{end_date}.csv".replace("-", "")
        )
        if cache_path.exists() and not refresh:
            frame = pd.read_csv(cache_path)
            if "Date" in frame.columns:
                frame["Date"] = pd.to_datetime(frame["Date"])
            return frame
        url = (
            f"{self.base_url}/v2/aggs/ticker/{quote(contract_ticker, safe='')}"
            f"/range/1/day/{start_date}/{end_date}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
        )
        raw_bars = self._fetch_paginated(url)
        frame = self._parse_results(raw_bars)
        frame["Date"] = frame["Date"].dt.normalize()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _write_csv_atomic(frame, cache_path)
        return frame

    def fetch_option_snapshot(self, underlying: str, contract_ticker: str) -> dict:
        """Fetch real-time snapshot with Greeks for a single option contract."""
        url = (
            f"{self.base_url}/v3/snapshot/options/{quote(underlying, safe='')}"
            f"/{quote(contract_ticker, safe='')}?apiKey={self.api_key}"
        )
        response = self._request_with_retry(url)
        payload = self._decode_response(response)
        results = payload.get("results")
        if not isinstance(results, dict):
            raise MarketDataError("Polygon snapshot response omitted a valid results object")
        return results


class FrozenMarketDataSource(MarketDataSource):
    """Read immutable normalized daily bars from CSV or Parquet snapshots."""

    _DATA_FILENAMES = ("ohlcv.parquet", "ohlcv.pq", "ohlcv.csv", "ohlcv.csv.gz")

    def __init__(
        self,
        source: str | Path,
        provenance: MarketDataProvenance | None = None,
    ) -> None:
        self.source = Path(source).expanduser()
        if not self.source.exists():
            raise MarketDataError(f"frozen market-data source does not exist: {self.source}")
        manifest_path = (
            self.source / "market_data_manifest.json"
            if self.source.is_dir()
            else self.source.parent / "market_data_manifest.json"
        )
        self._manifest: dict[str, Any] = {}
        manifest_provenance = None
        if manifest_path.exists():
            try:
                self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_provenance = MarketDataProvenance.from_record(
                    self._manifest["provenance"]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MarketDataError(f"invalid market-data manifest: {manifest_path}") from exc
        if (
            provenance is not None
            and manifest_provenance is not None
            and provenance != manifest_provenance
        ):
            raise MarketDataError(
                "explicit frozen-data provenance does not match market_data_manifest.json"
            )
        self._provenance = provenance or manifest_provenance
        if self._provenance is None:
            raise ValueError(
                "Frozen data requires market_data_manifest.json or explicit provider/feed/"
                "adjustment provenance"
            )
        self._loaded: dict[Path, pd.DataFrame] = {}

    @property
    def provenance(self) -> MarketDataProvenance:
        return self._provenance

    def _resolve_path(self, symbol: str) -> Path:
        if self.source.is_file():
            return self.source
        for name in self._DATA_FILENAMES:
            candidate = self.source / name
            if candidate.exists():
                return candidate
        for suffix in (".parquet", ".pq", ".csv", ".csv.gz"):
            for stem in (symbol, f"{symbol}_1d"):
                candidate = self.source / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
        raise MarketDataError(f"{symbol}: no frozen CSV/Parquet file under {self.source}")

    @staticmethod
    def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        canonical = {
            "date": "Date",
            "timestamp": "Date",
            "symbol": "Symbol",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "provider": "Provider",
            "feed": "Feed",
            "adjustment": "Adjustment",
            "interval": "Interval",
            "session": "Session",
        }
        renames = {
            column: canonical[str(column).strip().lower()]
            for column in frame.columns
            if str(column).strip().lower() in canonical
        }
        return frame.rename(columns=renames)

    def _read(self, path: Path, *, refresh: bool) -> pd.DataFrame:
        if path in self._loaded and not refresh:
            return self._loaded[path].copy()
        self._verify_snapshot(path)
        try:
            if path.suffix.lower() in {".parquet", ".pq"}:
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_csv(path)
        except ImportError as exc:
            raise MarketDataError(
                "Parquet support requires the optional data dependency: pip install -e '.[data]'"
            ) from exc
        except (OSError, ValueError) as exc:
            raise MarketDataError(f"cannot read frozen market data: {path}") from exc
        frame = self._canonicalize_columns(frame)
        self._validate_embedded_provenance(frame)
        self._loaded[path] = frame.copy()
        return frame

    def _verify_snapshot(self, path: Path) -> None:
        snapshot = self._manifest.get("snapshot") or {}
        if not isinstance(snapshot, dict) or snapshot.get("file") != path.name:
            return
        expected = str(snapshot.get("sha256") or "").strip().lower()
        if not expected:
            return
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise MarketDataError(
                f"frozen snapshot checksum does not match market_data_manifest.json: {path}"
            )

    def _validate_embedded_provenance(self, frame: pd.DataFrame) -> None:
        expected = {
            "Provider": self.provenance.provider,
            "Feed": self.provenance.feed,
            "Adjustment": self.provenance.adjustment,
            "Interval": self.provenance.interval,
            "Session": self.provenance.session,
        }
        for column, expected_value in expected.items():
            if column not in frame:
                continue
            values = {
                str(value).strip().lower()
                for value in frame[column].dropna().unique()
                if str(value).strip()
            }
            if len(values) > 1:
                raise MarketDataError(
                    f"frozen data mixes {column.lower()} values: {sorted(values)}"
                )
            if values and values != {expected_value}:
                raise MarketDataError(
                    f"frozen {column.lower()} {next(iter(values))!r} does not match "
                    f"declared {expected_value!r}"
                )

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
        path = self._resolve_path(symbol)
        raw = self._read(path, refresh=refresh)
        if "Symbol" in raw:
            raw = raw.loc[raw["Symbol"].astype(str).str.upper().eq(symbol)].copy()
        if raw.empty:
            raise MarketDataError(f"{symbol}: no frozen rows in source {path}")
        data = normalize_ohlcv(raw, symbol)
        data = data[(data["Date"] >= start) & (data["Date"] <= end)].reset_index(drop=True)
        if data.empty:
            raise MarketDataError(f"{symbol}: no frozen rows in requested date range")
        assert_not_stale(data, symbol, end, provider="frozen source")
        report = validate_ohlcv(data, symbol)
        if not report.ok:
            raise MarketDataError(f"{symbol}: frozen OHLCV validation failed: {report.as_record()}")
        retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        manifest_symbol = (self._manifest.get("symbols") or {}).get(symbol, {})
        if isinstance(manifest_symbol, dict) and manifest_symbol.get("retrieved_at"):
            retrieved_at = datetime.fromisoformat(
                str(manifest_symbol["retrieved_at"]).replace("Z", "+00:00")
            )
        return MarketDataResult(
            symbol=symbol,
            frame=data,
            provenance=self.provenance,
            retrieved_at=retrieved_at,
            cache_hit=True,
            metadata={"source_path": str(path.resolve()), "frozen": True},
        )
