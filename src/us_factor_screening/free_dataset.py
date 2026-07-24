"""Free Nasdaq-100 universe and aggregate Cboe option-volume acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request

import pandas as pd

from .data import (
    MarketDataError,
    MarketDataPanel,
    MarketDataProvenance,
    validate_us_symbol,
)
from .providers import HttpResponse, HttpTransport, urllib_transport

NASDAQ_100_ENDPOINT = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
CBOE_OPTION_VOLUME_ENDPOINT = (
    "https://www.cboe.com/us/options/market_statistics/historical_data/download/all_symbols/"
)
CBOE_EXCHANGES = ("CBOE", "BATS", "C2", "EDGX")
_CBOE_COLUMNS = (
    "Trade Date",
    "Options Class",
    "Underlying",
    "Product Type",
    "Exchange",
    "Volume",
)


@dataclass(frozen=True)
class Nasdaq100Snapshot:
    """One dated snapshot of the Nasdaq-100 constituent list."""

    as_of: pd.Timestamp
    constituents: pd.DataFrame

    @property
    def symbols(self) -> list[str]:
        return self.constituents["symbol"].tolist()


def _decode_json(response: HttpResponse, provider: str) -> dict[str, Any]:
    if response.status == 429:
        raise MarketDataError(f"{provider} rate limit reached")
    if response.status >= 400:
        detail = response.body.decode("utf-8", errors="replace")[:300].strip()
        raise MarketDataError(
            f"{provider} returned HTTP {response.status}: {detail or 'empty response'}"
        )
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketDataError(f"{provider} returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise MarketDataError(f"{provider} returned a non-object JSON payload")
    return payload


def fetch_nasdaq100_snapshot(
    *,
    endpoint: str = NASDAQ_100_ENDPOINT,
    timeout: float = 30.0,
    transport: HttpTransport = urllib_transport,
) -> Nasdaq100Snapshot:
    """Fetch the official current Nasdaq-100 constituent snapshot."""

    response = transport(
        Request(
            endpoint,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 us-factor-screening/0.1",
            },
        ),
        timeout,
    )
    payload = _decode_json(response, "Nasdaq")
    try:
        data = payload["data"]
        rows = data["data"]["rows"]
        as_of = pd.Timestamp(data["date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MarketDataError("Nasdaq response omitted constituent rows or snapshot date") from exc
    if not isinstance(rows, list) or not rows:
        raise MarketDataError("Nasdaq returned an empty constituent list")

    records = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            symbol = validate_us_symbol(str(item["symbol"]))
        except (KeyError, ValueError):
            continue
        records.append(
            {
                "symbol": symbol,
                "company_name": str(item.get("companyName") or "").strip(),
                "snapshot_date": as_of.normalize(),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise MarketDataError("Nasdaq returned no usable constituent symbols")
    if frame["symbol"].duplicated().any():
        duplicates = sorted(frame.loc[frame["symbol"].duplicated(), "symbol"].unique())
        raise MarketDataError(f"Nasdaq returned duplicate symbols: {duplicates}")
    frame = frame.sort_values("symbol").reset_index(drop=True)
    return Nasdaq100Snapshot(as_of=as_of.normalize(), constituents=frame)


class CboeOptionVolumeSource:
    """Daily aggregate option volume from Cboe's four public venue reports."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/cboe_option_volume",
        *,
        endpoint: str = CBOE_OPTION_VOLUME_ENDPOINT,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        transport: HttpTransport = urllib_transport,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay cannot be negative")
        self.cache_dir = Path(cache_dir)
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.transport = transport
        self.sleeper = sleeper

    def _cache_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"cboe_option_volume_{year:04d}_{month:02d}.csv"

    def _url(self, year: int, month: int) -> str:
        query = urlencode(
            {
                "reportType": "volume",
                "month": str(month),
                "year": str(year),
                "volumeType": "sum",
                "volumeAggType": "daily",
                "exchanges": list(CBOE_EXCHANGES),
            },
            doseq=True,
        )
        return f"{self.endpoint}?{query}"

    def _download(self, year: int, month: int) -> bytes:
        url = self._url(year, month)
        for attempt in range(self.max_retries + 1):
            response = self.transport(
                Request(
                    url,
                    headers={
                        "Accept": "text/csv",
                        "User-Agent": "Mozilla/5.0 us-factor-screening/0.1",
                    },
                ),
                self.timeout,
            )
            if response.status == 429 or response.status >= 500:
                if attempt >= self.max_retries:
                    raise MarketDataError(
                        f"Cboe option-volume request failed with HTTP {response.status}"
                    )
                self.sleeper(self.retry_base_delay * (2**attempt))
                continue
            if response.status >= 400:
                detail = response.body.decode("utf-8", errors="replace")[:300].strip()
                raise MarketDataError(
                    f"Cboe option-volume request returned HTTP {response.status}: "
                    f"{detail or 'empty response'}"
                )
            if response.body.lstrip().lower().startswith(b"<!doctype html"):
                raise MarketDataError("Cboe option-volume request returned HTML instead of CSV")
            return response.body
        raise AssertionError("unreachable")

    def fetch_month(self, year: int, month: int, *, refresh: bool = False) -> pd.DataFrame:
        if year < 2007:
            raise ValueError("Cboe historical option volume starts in 2007")
        if month not in range(1, 13):
            raise ValueError("month must be between 1 and 12")
        path = self._cache_path(year, month)
        if path.exists() and not refresh:
            raw = path.read_bytes()
        else:
            raw = self._download(year, month)
            _write_bytes_atomic(raw, path)
        try:
            frame = pd.read_csv(BytesIO(raw))
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise MarketDataError("Cboe returned malformed option-volume CSV") from exc
        missing = set(_CBOE_COLUMNS).difference(frame.columns)
        if missing:
            raise MarketDataError(f"Cboe option-volume CSV is missing columns: {sorted(missing)}")
        return _normalize_cboe_volume(frame)

    def fetch_range(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            raise ValueError("start_date must be on or before end_date")
        selected = {validate_us_symbol(symbol) for symbol in symbols}
        if not selected:
            raise ValueError("At least one symbol is required")

        months = pd.period_range(start=start, end=end, freq="M")
        frames = [
            self.fetch_month(period.year, period.month, refresh=refresh)
            for period in months
        ]
        data = pd.concat(frames, ignore_index=True)
        data = data.loc[
            data["Symbol"].isin(selected)
            & data["Date"].between(start, end, inclusive="both")
        ].copy()
        if data.empty:
            raise MarketDataError("Cboe returned no option-volume rows for the selected universe")
        return data.sort_values(
            ["Date", "Symbol", "OptionRoot", "Exchange"]
        ).reset_index(drop=True)


def _normalize_cboe_volume(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.loc[:, _CBOE_COLUMNS].rename(
        columns={
            "Trade Date": "Date",
            "Options Class": "OptionRoot",
            "Underlying": "Symbol",
            "Product Type": "ProductType",
            "Volume": "OptionVolume",
        }
    )
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in ("OptionRoot", "Symbol", "ProductType", "Exchange"):
        data[column] = data[column].astype(str).str.strip().str.upper()
    data["OptionVolume"] = pd.to_numeric(data["OptionVolume"], errors="coerce")
    data = data.dropna(subset=["Date", "OptionVolume"])
    if (data["OptionVolume"] < 0).any():
        raise MarketDataError("Cboe option volume contains negative values")
    if not set(data["Exchange"]).issubset(CBOE_EXCHANGES):
        unexpected = sorted(set(data["Exchange"]).difference(CBOE_EXCHANGES))
        raise MarketDataError(f"Cboe option volume contains unknown exchanges: {unexpected}")
    data["OptionVolume"] = data["OptionVolume"].astype("int64")
    return data.reset_index(drop=True)


def aggregate_cboe_option_volume(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate option roots and Cboe venues to one row per date and underlying."""

    required = {"Date", "Symbol", "OptionRoot", "Exchange", "OptionVolume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"option-volume frame is missing columns: {sorted(missing)}")
    grouped = frame.groupby(["Date", "Symbol"], sort=True)
    result = grouped.agg(
        OptionVolume=("OptionVolume", "sum"),
        OptionRootCount=("OptionRoot", "nunique"),
        ReportingExchangeCount=("Exchange", "nunique"),
    ).reset_index()
    result["Provider"] = "cboe_public_statistics"
    result["Coverage"] = "cboe_venues_only"
    return result


def select_price_history_eligible(
    panel: MarketDataPanel,
    *,
    requested_start: str,
    minimum_sessions: int = 252,
    maximum_start_gap_days: int = 10,
) -> tuple[MarketDataPanel, pd.DataFrame]:
    """Exclude symbols that lack enough history at the requested window's start."""

    if minimum_sessions < 1:
        raise ValueError("minimum_sessions must be at least 1")
    if maximum_start_gap_days < 0:
        raise ValueError("maximum_start_gap_days cannot be negative")
    start = pd.Timestamp(requested_start).normalize()
    latest_acceptable_start = start + pd.Timedelta(days=maximum_start_gap_days)
    records: list[dict[str, Any]] = []
    eligible_results = {}
    for symbol, result in panel.results.items():
        frame = result.frame
        first_date = frame["Date"].min().normalize()
        enough_sessions = len(frame) >= minimum_sessions
        present_at_start = first_date <= latest_acceptable_start
        reasons = []
        if not present_at_start:
            reasons.append("history_starts_after_window_tolerance")
        if not enough_sessions:
            reasons.append("insufficient_price_sessions")
        eligible = not reasons
        if eligible:
            eligible_results[symbol] = result
        records.append(
            {
                "symbol": symbol,
                "price_sessions": len(frame),
                "first_price_date": first_date,
                "last_price_date": frame["Date"].max().normalize(),
                "eligible": eligible,
                "exclusion_reason": ";".join(reasons),
            }
        )
    if not eligible_results:
        raise MarketDataError("No symbols passed the price-history eligibility rules")
    return MarketDataPanel(eligible_results), pd.DataFrame.from_records(records)


def write_free_dataset_bundle(
    output_dir: str | Path,
    *,
    universe: pd.DataFrame,
    option_detail: pd.DataFrame,
    option_daily: pd.DataFrame,
    start_date: str,
    end_date: str,
    universe_basis: str,
    minimum_price_sessions: int = 252,
    maximum_start_gap_days: int = 10,
    archive: str | Path | None = None,
    price_provenance: MarketDataProvenance | None = None,
) -> Path | None:
    """Write option/universe artifacts and a checksum manifest beside OHLCV outputs."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "universe": target / "nasdaq100_universe.csv",
        "option_detail": target / "cboe_option_volume_by_venue.csv.gz",
        "option_daily": target / "cboe_option_volume_daily.csv.gz",
    }
    universe.to_csv(paths["universe"], index=False)
    option_detail.to_csv(paths["option_detail"], index=False, compression="gzip")
    option_daily.to_csv(paths["option_daily"], index=False, compression="gzip")

    included_mask = (
        universe["eligible"].fillna(False).astype(bool)
        if "eligible" in universe
        else pd.Series(True, index=universe.index)
    )
    for optional_name in ("ohlcv.csv", "data_quality.csv", "market_data_manifest.json"):
        optional_path = target / optional_name
        if optional_path.exists():
            paths[optional_name.removesuffix(".json").removesuffix(".csv")] = optional_path

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "period": {"start": str(start_date), "end": str(end_date)},
        "universe": {
            "basis": universe_basis,
            "symbols_in_source": int(universe["symbol"].nunique()),
            "symbols_included": int(universe.loc[included_mask, "symbol"].nunique()),
            "eligibility": {
                "rule": "price history begins near requested start and meets minimum sessions",
                "minimum_price_sessions": minimum_price_sessions,
                "maximum_start_gap_days": maximum_start_gap_days,
                "warning": (
                    "This is a data-history proxy for excluding recent listings. It does not "
                    "measure public float or shares available to trade."
                ),
            },
            "warning": (
                "A current constituent snapshot is not point-in-time membership and creates "
                "survivorship bias when applied to earlier dates."
                if universe_basis == "current_nasdaq100_snapshot"
                else None
            ),
        },
        "price_data": {
            "provider": price_provenance.provider if price_provenance else "yahoo_finance",
            "feed": price_provenance.feed if price_provenance else "yahoo_chart",
            "granularity": "daily OHLCV",
            "adjustment": (
                price_provenance.adjustment if price_provenance else "backward_total_return"
            ),
            "corporate_actions": "splits_and_cash_distributions",
            "limitations": [
                (
                    f"{price_provenance.provider if price_provenance else 'yahoo_finance'} "
                    "data is suitable for research prototyping, not an official exchange record."
                ),
                "Volume is reported stock volume; price adjustment does not turn it into option volume.",
            ],
        },
        "option_data": {
            "provider": "cboe_public_statistics",
            "endpoint": CBOE_OPTION_VOLUME_ENDPOINT,
            "venues": list(CBOE_EXCHANGES),
            "measurement": "executed option contract volume",
            "granularity": "daily underlying/option-root/venue",
            "symbols_with_reported_volume": int(option_detail["Symbol"].nunique()),
            "limitations": [
                "Cboe venues only; this is not consolidated OPRA volume.",
                "No strike, expiration, call/put, quote, open interest, IV, or Greeks.",
                "A missing row can mean zero volume or no applicable listing.",
            ],
        },
        "files": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = target / "free_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    if archive is None:
        return None
    archive_path = Path(archive)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    included = [*paths.values(), manifest_path]
    with tarfile.open(archive_path, "w:gz") as bundle:
        for path in sorted(included, key=lambda item: item.name):
            bundle.add(path, arcname=f"free_nasdaq100/{path.name}")
    return archive_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
