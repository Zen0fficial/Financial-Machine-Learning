#!/usr/bin/env python3
"""Standalone Polygon database builder.

Downloads daily + intraday stock bars, option contract references, option
daily aggregates, and real-time option snapshots (with Greeks) for the
Nasdaq-100 universe. Designed to run unattended on a server: resumable,
concurrent, fault-tolerant, with rolling logs and a final manifest.

Usage:
    POLYGON_API_KEY=... python scripts/download_database.py \\
        --data-dir data/polygon_db \\
        --start 2024-01-02 --end 2026-07-23

    # Or pass the key explicitly:
    python scripts/download_database.py --api-key ma_... \\
        --data-dir data/polygon_db --minute --workers 8

Phases (each skippable via --skip-<phase>):
    1. daily               — daily OHLCV+VWAP for all symbols
    2. minute              — 1-minute OHLCV+VWAP (large; ~2.4 GB uncompressed)
    3. options-contracts   — option contract reference (strikes/expirations)
    4. options-aggregates  — daily OHLCV for selected contracts (ATM +/-5,
                             front 3 expirations per symbol)
    5. options-snapshots   — real-time Greeks/IV/OI for selected contracts
    6. financials          — quarterly fundamentals (balance sheet + income +
                             cash flow; ~80+ fields per quarter)
    7. news                — article metadata per ticker (title, description,
                             publisher, keywords; no sentiment scores)
    8. splits              — stock split history
    9. dividends           — dividend history
   10. ticker-details      — company info (name, market cap, SIC, etc.)
   11. open-close          — daily pre/post market prices (1 request per symbol
                             per day; ~62000 requests for 97 symbols × 639 days;
                             OFF by default, enable with --open-close)

The script is resumable: cached files are skipped, failed symbols are
retried on the next run, and a manifest.json records the final state.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure the project's src/ is importable when run as a standalone script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from us_factor_screening.data import MarketDataError  # noqa: E402
from us_factor_screening.providers import PolygonMarketDataSource  # noqa: E402

# Production runs must pass --universe-file for the intended Nasdaq-100 or
# S&P 500 member history. There is deliberately no smoke-test fallback.

SYMBOL_COLUMNS = ("Symbol", "symbol", "Ticker", "ticker")
LISTED_DATE_COLUMNS = (
    "listed_date", "list_date", "listing_date", "ipo_date", "start_date",
    "index_start", "membership_start",
)
DELISTED_DATE_COLUMNS = (
    "delisted_date", "delist_date", "delisting_date", "end_date",
    "index_end", "membership_end",
)
ACTIVE_COLUMNS = ("active", "is_active")
STATUS_COLUMNS = ("status", "listing_status")
SOURCE_EXCLUSION_COLUMNS = ("exclusion_reason", "source_exclusion_reason")


def _symbol_values(value: str | None) -> set[str]:
    """Resolve comma-separated symbols or a one-column CSV into a symbol set."""
    if not value:
        return set()
    if Path(value).exists():
        frame = pd.read_csv(value)
        for col in ("Symbol", "symbol", "Ticker", "ticker"):
            if col in frame.columns:
                return {str(s).strip().upper() for s in frame[col] if str(s).strip()}
        return {str(s).strip().upper() for s in frame.iloc[:, 0] if str(s).strip()}
    return {s.strip().upper() for s in value.split(",") if s.strip()}


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    exact = {str(col): str(col) for col in frame.columns}
    lowered = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _parse_optional_dates(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(pd.NaT, index=index)
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _format_optional_dates(values: pd.Series) -> pd.Series:
    return values.dt.strftime("%Y-%m-%d").fillna("")


def _default_universe_candidates(universe: str) -> list[Path]:
    candidates = [
        _PROJECT_ROOT / "data" / "universes" / f"{universe}.csv",
        _PROJECT_ROOT / "data" / f"{universe}_universe.csv",
    ]
    if universe == "nasdaq100":
        candidates.extend([
            _PROJECT_ROOT / "data" / "free_nasdaq100_2024_2026" / "nasdaq100_universe.csv",
            _PROJECT_ROOT / "data" / "nasdaq100_universe.csv",
        ])
    return candidates


def _load_universe_source(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    universe_file = getattr(args, "universe_file", None)
    if universe_file:
        path = Path(universe_file)
        return pd.read_csv(path), str(path)

    symbols_arg = getattr(args, "symbols", "all")
    if symbols_arg and str(symbols_arg).lower() != "all":
        path = Path(str(symbols_arg))
        if path.exists():
            return pd.read_csv(path), str(path)
        return pd.DataFrame({"symbol": sorted(_symbol_values(str(symbols_arg)))}), "cli-symbols"

    universe = str(getattr(args, "universe", "custom"))
    for candidate in _default_universe_candidates(universe):
        if candidate.exists():
            return pd.read_csv(candidate), str(candidate)

    raise FileNotFoundError(
        f"No universe file found for '{universe}'. Pass --universe-file with a CSV "
        "containing at least a symbol/ticker column and optional lifecycle fields."
    )


def build_universe_table(args: argparse.Namespace) -> pd.DataFrame:
    """Build a normalized universe table with listing lifecycle fields."""
    frame, source = _load_universe_source(args)
    symbol_col = _find_column(frame, SYMBOL_COLUMNS)
    if symbol_col is None:
        raise ValueError("Universe source must contain a symbol/ticker column")

    table = pd.DataFrame({"symbol": frame[symbol_col].astype(str).str.strip().str.upper()})
    table = table[table["symbol"] != ""].drop_duplicates("symbol").reset_index(drop=True)
    frame = frame.loc[table.index].reset_index(drop=True)

    listed_col = _find_column(frame, LISTED_DATE_COLUMNS)
    delisted_col = _find_column(frame, DELISTED_DATE_COLUMNS)
    active_col = _find_column(frame, ACTIVE_COLUMNS)
    status_col = _find_column(frame, STATUS_COLUMNS)
    source_exclusion_col = _find_column(frame, SOURCE_EXCLUSION_COLUMNS)

    listed_dates = _parse_optional_dates(frame[listed_col] if listed_col else None, table.index)
    delisted_dates = _parse_optional_dates(frame[delisted_col] if delisted_col else None, table.index)
    period_start = pd.Timestamp(getattr(args, "listed_since", "2024-01-01")).normalize()
    period_end = pd.Timestamp(getattr(args, "end", datetime.now(UTC).strftime("%Y-%m-%d"))).normalize()
    explicit_excluded = _symbol_values(getattr(args, "exclude_symbols", None))

    listed_during_period = (listed_dates.isna() | (listed_dates <= period_end)) & (
        delisted_dates.isna() | (delisted_dates >= period_start)
    )
    explicitly_excluded = table["symbol"].isin(explicit_excluded)

    source_exclusion = (
        frame[source_exclusion_col].fillna("").astype(str).str.strip()
        if source_exclusion_col
        else pd.Series("", index=table.index)
    )
    respect_source_exclusions = bool(getattr(args, "respect_source_exclusions", False))
    source_excluded = source_exclusion.ne("") & respect_source_exclusions

    reasons: list[str] = []
    for idx in range(len(table)):
        symbol_reasons: list[str] = []
        if not bool(listed_during_period.iloc[idx]):
            symbol_reasons.append("not_listed_during_period")
        if bool(explicitly_excluded.iloc[idx]):
            symbol_reasons.append("explicitly_excluded")
        if bool(source_excluded.iloc[idx]):
            symbol_reasons.append(str(source_exclusion.iloc[idx]))
        reasons.append(";".join(symbol_reasons))

    if active_col:
        active = frame[active_col].map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})
    else:
        active = pd.Series(pd.NA, index=table.index)
    status = frame[status_col].fillna("").astype(str) if status_col else pd.Series("", index=table.index)

    table["universe"] = getattr(args, "universe", "custom")
    table["universe_source"] = source
    table["listed_since"] = period_start.strftime("%Y-%m-%d")
    table["listed_until"] = period_end.strftime("%Y-%m-%d")
    table["listed_date"] = _format_optional_dates(listed_dates)
    table["delisted_date"] = _format_optional_dates(delisted_dates)
    table["active"] = active
    table["source_status"] = status
    table["source_exclusion_reason"] = source_exclusion
    table["listed_during_period"] = listed_during_period
    table["listed_after_period_start"] = listed_dates.notna() & (listed_dates > period_start) & (listed_dates <= period_end)
    table["delisted_during_period"] = (
        delisted_dates.notna() & (delisted_dates >= period_start) & (delisted_dates <= period_end)
    )
    table["included"] = listed_during_period & ~explicitly_excluded & ~source_excluded
    table["exclusion_reason"] = reasons
    return table.sort_values("symbol").reset_index(drop=True)


def build_symbol_windows(
    universe: pd.DataFrame, start: str, end: str
) -> dict[str, tuple[str, str]]:
    """Clamp the requested range to each included symbol's listing lifecycle."""
    requested_start = pd.Timestamp(start).normalize()
    requested_end = pd.Timestamp(end).normalize()
    windows: dict[str, tuple[str, str]] = {}
    for row in universe.loc[universe["included"]].itertuples(index=False):
        symbol_start = requested_start
        symbol_end = requested_end
        listed_date = pd.to_datetime(row.listed_date, errors="coerce")
        delisted_date = pd.to_datetime(row.delisted_date, errors="coerce")
        if not pd.isna(listed_date):
            symbol_start = max(symbol_start, listed_date.normalize())
        if not pd.isna(delisted_date):
            symbol_end = min(symbol_end, delisted_date.normalize())
        if symbol_start <= symbol_end:
            windows[str(row.symbol)] = (
                symbol_start.strftime("%Y-%m-%d"),
                symbol_end.strftime("%Y-%m-%d"),
            )
    return windows


def build_daily_coverage_windows(
    symbols: list[str],
    symbol_windows: dict[str, tuple[str, str]],
    daily_cache_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, tuple[str, str]]:
    """Move each symbol window start to its first actual daily row.

    Lifecycle metadata can be broader than the provider's effective coverage
    for a ticker. Later date-sensitive phases should avoid pre-coverage starts
    that Polygon may report as malformed/no-results responses, while preserving
    the requested/lifecycle end date so resumed downloads keep extending forward.
    """
    adjusted = dict(symbol_windows)
    for symbol in symbols:
        if symbol not in symbol_windows:
            continue
        matches = sorted(daily_cache_dir.glob(f"polygon_1d_*_{symbol}_*.csv"))
        if not matches:
            continue

        dates: list[pd.Timestamp] = []
        for path in matches:
            try:
                frame = pd.read_csv(path, usecols=["Date"])
            except Exception:  # noqa: BLE001
                continue
            parsed = pd.to_datetime(frame["Date"], errors="coerce").dropna()
            if not parsed.empty:
                dates.extend(parsed.dt.normalize().tolist())
        if not dates:
            continue

        planned_start, planned_end = symbol_windows[symbol]
        start_ts = max(pd.Timestamp(planned_start), min(dates))
        end_ts = pd.Timestamp(planned_end)
        if start_ts > end_ts:
            continue

        new_window = (start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        if new_window != symbol_windows[symbol]:
            adjusted[symbol] = new_window
            if logger:
                logger.info(
                    "  %s: adjusted start date %s → %s based on first daily row",
                    symbol,
                    planned_start,
                    new_window[0],
                )
    return adjusted


def load_symbols(args: argparse.Namespace) -> list[str]:
    """Resolve included symbols from the normalized universe table."""
    universe = build_universe_table(args)
    return universe.loc[universe["included"], "symbol"].astype(str).tolist()


def setup_logging(log_file: Path | None) -> logging.Logger:
    """Configure logging to both console and (optionally) a file."""
    logger = logging.getLogger("polygon_dl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def fetch_symbol_daily(
    source: PolygonMarketDataSource, start: str, end: str, symbol: str, logger: logging.Logger
) -> tuple[bool, str]:
    """Download daily aggregates for one symbol. Returns (ok, message)."""
    try:
        result = source.fetch_result(symbol, start, end, refresh=False)
        rows = len(result.frame)
        if rows == 0:
            return False, f"{symbol}: no rows"
        return True, f"{symbol}: {rows} rows ({result.frame['Date'].min().date()} → {result.frame['Date'].max().date()})"
    except MarketDataError as exc:
        return False, f"{symbol}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: unexpected error: {exc}"


def fetch_symbol_minute(
    source: PolygonMarketDataSource, start: str, end: str, symbol: str, logger: logging.Logger
) -> tuple[bool, str]:
    """Download 1-minute aggregates for one symbol. Returns (ok, message)."""
    try:
        # Polygon caps minute aggregates at 50,000 bars per request (~166 days).
        # Split the range into quarterly chunks to stay under the cap and to
        # give resumable checkpoints.
        chunks = _quarterly_chunks(start, end)
        total = 0
        for chunk_start, chunk_end in chunks:
            result = source.fetch_result(symbol, chunk_start, chunk_end, refresh=False)
            total += len(result.frame)
        return True, f"{symbol}: {total} minute bars"
    except MarketDataError as exc:
        return False, f"{symbol}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: unexpected error: {exc}"


def _quarterly_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into ~90-day chunks for minute-aggregate pagination."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    chunks: list[tuple[str, str]] = []
    cursor = s
    while cursor <= e:
        nxt = min(cursor + pd.Timedelta(days=90), e)
        chunks.append((cursor.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cursor = nxt + pd.Timedelta(days=1)
    return chunks


def fetch_option_contracts(
    source: PolygonMarketDataSource, symbol: str, logger: logging.Logger
) -> tuple[bool, str, pd.DataFrame | None]:
    """Fetch option contract reference for one underlying."""
    try:
        frame = source.fetch_option_contracts(symbol, refresh=False)
        return True, f"{symbol}: {len(frame)} contracts", frame
    except MarketDataError as exc:
        return False, f"{symbol}: {exc}", None
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: unexpected error: {exc}", None


def select_contracts_for_aggregates(
    contracts: pd.DataFrame, spot: float | None, logger: logging.Logger
) -> list[str]:
    """Select a tractable subset of contracts for historical aggregate download.

    Heuristic: for each expiration, pick the 5 calls and 5 puts whose strikes
    are closest to the spot (or to 1.0 if spot is unknown). Keep only the
    front 6 monthly expirations to bound the total count.
    """
    if contracts is None or contracts.empty:
        return []
    df = contracts.copy()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    df = df.dropna(subset=["expiration_date"])
    # Keep expirations >= today and sort ascending; take the front 6.
    today = pd.Timestamp.now(tz=None).normalize()
    df = df[df["expiration_date"] >= today].sort_values("expiration_date")
    expirations = df["expiration_date"].drop_duplicates().head(6)
    df = df[df["expiration_date"].isin(expirations)]
    ref = spot if spot and spot > 0 else 1.0
    selected: list[str] = []
    for _expiry, group in df.groupby("expiration_date"):
        for ctype in ("call", "put"):
            sub = group[group["contract_type"] == ctype]
            if sub.empty:
                continue
            sub = sub.assign(_dist=(sub["strike_price"] - ref).abs())
            selected.extend(sub.sort_values("_dist").head(5)["ticker"].tolist())
    return selected


def fetch_option_aggregates_for_symbol(
    source: PolygonMarketDataSource,
    start: str,
    end: str,
    symbol: str,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """Download daily aggregates for a selected set of option contracts."""
    try:
        ok, msg, contracts = fetch_option_contracts(source, symbol, logger)
        if not ok or contracts is None or contracts.empty:
            return ok, msg
        # Estimate spot from the most recent daily bar.
        spot: float | None = None
        try:
            daily = source.fetch_result(symbol, start, end, refresh=False)
            if not daily.frame.empty:
                spot = float(daily.frame["Close"].iloc[-1])
        except Exception:  # noqa: BLE001
            pass
        selected = select_contracts_for_aggregates(contracts, spot, logger)
        if not selected:
            return True, f"{symbol}: no selectable option contracts"
        successes = 0
        failures = 0
        for contract in selected:
            try:
                frame = source.fetch_option_aggregates(contract, start, end, refresh=False)
                if not frame.empty:
                    successes += 1
                else:
                    failures += 1
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.debug(f"{contract}: {exc}")
        return True, f"{symbol}: {successes}/{len(selected)} option contracts downloaded ({failures} failed)"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: unexpected error: {exc}"


def fetch_option_snapshots_for_symbol(
    source: PolygonMarketDataSource, symbol: str, logger: logging.Logger
) -> tuple[bool, str]:
    """Fetch real-time snapshots (with Greeks) for a selected set of contracts."""
    try:
        ok, msg, contracts = fetch_option_contracts(source, symbol, logger)
        if not ok or contracts is None or contracts.empty:
            return ok, msg
        # Reuse the same selection logic; spot is fetched live in the snapshot.
        selected = select_contracts_for_aggregates(contracts, None, logger)
        if not selected:
            return True, f"{symbol}: no selectable option contracts for snapshot"
        snapshot_dir = source.cache_dir / "option_snapshots" / datetime.now(UTC).strftime("%Y-%m-%d")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        successes = 0
        for contract in selected:
            try:
                payload = source.fetch_option_snapshot(symbol, contract)
                out = snapshot_dir / f"{contract.replace(':', '_')}.json"
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                successes += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"{contract} snapshot: {exc}")
        return True, f"{symbol}: {successes}/{len(selected)} snapshots saved"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: unexpected error: {exc}"


# Lock guarding concurrent updates to the shared ticker_details.csv.
_TICKER_DETAILS_LOCK = threading.Lock()


def _fetch_single(source: PolygonMarketDataSource, url: str) -> dict[str, Any]:
    """Perform a single (non-paginated) HTTP request and return parsed JSON."""
    url = source._append_api_key(url)
    response = source._request_with_retry(url)
    return source._decode_response(response)


def fetch_financials(
    source: PolygonMarketDataSource, symbol: str, logger: logging.Logger
) -> tuple[bool, str]:
    """Download quarterly fundamentals (balance sheet + income + cash flow)."""
    try:
        out_dir = source.cache_dir / "financials"
        out_path = out_dir / f"financials_{symbol}.csv"
        if out_path.exists():
            existing = pd.read_csv(out_path)
            return True, f"{symbol}: {len(existing)} quarters (cached)"
        url = (
            f"{source.base_url}/v2/reference/financials/{symbol}"
            f"?limit=100&apiKey={source.api_key}"
        )
        raw = source._fetch_paginated(url)
        if not raw:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(out_path, index=False)
            return True, f"{symbol}: 0 quarters"
        frame = pd.DataFrame.from_records(raw)
        # Sort by report_period descending (API rejects sort= param, so do it locally).
        if "report_period" in frame.columns:
            frame = frame.sort_values("report_period", ascending=False).reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        return True, f"{symbol}: {len(frame)} quarters"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: {exc}"


def fetch_news(
    source: PolygonMarketDataSource, symbol: str, logger: logging.Logger
) -> tuple[bool, str]:
    """Download news article metadata for one symbol."""
    columns = [
        "id", "published_utc", "title", "description",
        "publisher_name", "article_url", "tickers", "keywords",
    ]
    try:
        out_dir = source.cache_dir / "news"
        out_path = out_dir / f"news_{symbol}.csv"
        if out_path.exists():
            existing = pd.read_csv(out_path)
            return True, f"{symbol}: {len(existing)} articles (cached)"
        url = (
            f"{source.base_url}/v2/reference/news"
            f"?ticker={symbol}&limit=1000&sort=published_utc&order=desc&apiKey={source.api_key}"
        )
        raw = source._fetch_paginated(url)
        if not raw:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=columns).to_csv(out_path, index=False)
            return True, f"{symbol}: 0 articles"
        rows = []
        for item in raw:
            publisher = item.get("publisher") or {}
            rows.append({
                "id": item.get("id", ""),
                "published_utc": item.get("published_utc", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "publisher_name": publisher.get("name", "") if isinstance(publisher, dict) else "",
                "article_url": item.get("article_url", ""),
                "tickers": ";".join(item.get("tickers") or []),
                "keywords": ";".join(item.get("keywords") or []),
            })
        frame = pd.DataFrame(rows, columns=columns)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        return True, f"{symbol}: {len(frame)} articles"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: {exc}"


def fetch_splits(
    source: PolygonMarketDataSource, symbol: str, logger: logging.Logger
) -> tuple[bool, str]:
    """Download stock split history for one symbol."""
    try:
        out_dir = source.cache_dir / "splits"
        out_path = out_dir / f"splits_{symbol}.csv"
        if out_path.exists():
            existing = pd.read_csv(out_path)
            return True, f"{symbol}: {len(existing)} splits (cached)"
        url = (
            f"{source.base_url}/v3/reference/splits"
            f"?ticker={symbol}&limit=100&apiKey={source.api_key}"
        )
        raw = source._fetch_paginated(url)
        if not raw:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(out_path, index=False)
            return True, f"{symbol}: 0 splits"
        frame = pd.DataFrame.from_records(raw)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        return True, f"{symbol}: {len(frame)} splits"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: {exc}"


def fetch_dividends(
    source: PolygonMarketDataSource, symbol: str, logger: logging.Logger
) -> tuple[bool, str]:
    """Download dividend history for one symbol."""
    try:
        out_dir = source.cache_dir / "dividends"
        out_path = out_dir / f"dividends_{symbol}.csv"
        if out_path.exists():
            existing = pd.read_csv(out_path)
            return True, f"{symbol}: {len(existing)} dividends (cached)"
        url = (
            f"{source.base_url}/v3/reference/dividends"
            f"?ticker={symbol}&limit=100&apiKey={source.api_key}"
        )
        raw = source._fetch_paginated(url)
        if not raw:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(out_path, index=False)
            return True, f"{symbol}: 0 dividends"
        frame = pd.DataFrame.from_records(raw)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        return True, f"{symbol}: {len(frame)} dividends"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: {exc}"


def fetch_ticker_details(
    source: PolygonMarketDataSource,
    symbol: str,
    logger: logging.Logger,
    reference_date: str | None = None,
) -> tuple[bool, str]:
    """Download company info and merge it into a schema-aligned combined CSV."""
    try:
        out_path = source.cache_dir / "ticker_details.csv"
        with _TICKER_DETAILS_LOCK:
            if out_path.exists():
                existing = pd.read_csv(out_path)
                if "ticker" in existing.columns and symbol in existing["ticker"].astype(str).values:
                    match = existing.loc[existing["ticker"].astype(str) == symbol]
                    if not match.empty and "name" in match.columns:
                        name = str(match["name"].iloc[0])
                        return True, f"{symbol}: {name} (cached)"

        urls = [(None, f"{source.base_url}/v3/reference/tickers/{symbol}")]
        if reference_date:
            historical_date = (
                pd.Timestamp(reference_date) - pd.offsets.BusinessDay(1)
            ).strftime("%Y-%m-%d")
            urls.append((historical_date, f"{source.base_url}/v3/reference/tickers/{symbol}?date={historical_date}"))

        results: dict[str, Any] | None = None
        used_reference_date: str | None = None
        errors: list[str] = []
        for candidate_date, url in urls:
            try:
                payload = _fetch_single(source, url)
                candidate = payload.get("results")
                if isinstance(candidate, dict):
                    results = candidate
                    used_reference_date = candidate_date
                    break
                errors.append(f"{candidate_date or 'current'}: no results")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{candidate_date or 'current'}: {exc}")
        if results is None:
            return False, f"{symbol}: {'; '.join(errors)}"

        name = results.get("name") or symbol
        row = pd.json_normalize([results])
        row["reference_date"] = used_reference_date or ""
        with _TICKER_DETAILS_LOCK:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
            combined = pd.concat([existing, row], ignore_index=True, sort=False)
            combined = combined.drop_duplicates(subset=["ticker"], keep="last")
            combined = combined.sort_values("ticker").reset_index(drop=True)
            tmp_path = out_path.with_suffix(".csv.tmp")
            combined.to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)
        suffix = f" (as of {used_reference_date})" if used_reference_date else ""
        return True, f"{symbol}: {name}{suffix}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: {exc}"


def fetch_ticker_details_for_window(
    source: PolygonMarketDataSource,
    start: str,
    end: str,
    symbol: str,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """Fetch current details, falling back to the symbol lifecycle window."""
    del start
    return fetch_ticker_details(source, symbol, logger, reference_date=end)


def fetch_open_close(
    source: PolygonMarketDataSource,
    start: str,
    end: str,
    symbol: str,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """Download daily open/close (with pre/post market) for one symbol.

    This is 1 request per trading day per symbol — very expensive. Skips
    weekends only (no holiday calendar). Resumable: dates already in the
    cache file are not re-fetched.
    """
    try:
        out_dir = source.cache_dir / "open_close"
        out_path = out_dir / f"open_close_{symbol}_{start}_{end}.csv"
        trading_days = pd.bdate_range(start=start, end=end).strftime("%Y-%m-%d").tolist()

        existing_frame = pd.DataFrame()
        existing_dates: set[str] = set()
        if out_path.exists():
            existing_frame = pd.read_csv(out_path)
            if not existing_frame.empty and "from" in existing_frame.columns:
                existing_dates = set(existing_frame["from"].astype(str).tolist())

        missing_days = [d for d in trading_days if d not in existing_dates]
        if not missing_days:
            return True, f"{symbol}: {len(existing_frame)} days (cached)"

        logger.info(f"  {symbol}: fetching {len(missing_days)}/{len(trading_days)} open-close days")
        new_rows: list[dict[str, Any]] = []
        for date in missing_days:
            url = f"{source.base_url}/v1/open-close/{symbol}/{date}?apiKey={source.api_key}"
            try:
                payload = _fetch_single(source, url)
                if payload.get("status") != "OK":
                    continue
                new_rows.append(payload)
            except Exception:  # noqa: BLE001
                pass
            if source.inter_request_delay > 0.0:
                source.sleeper(source.inter_request_delay)

        new_frame = pd.DataFrame.from_records(new_rows) if new_rows else pd.DataFrame()
        if not existing_frame.empty and not new_frame.empty:
            frame = pd.concat([existing_frame, new_frame], ignore_index=True)
        elif not new_frame.empty:
            frame = new_frame
        else:
            frame = existing_frame

        if frame.empty:
            return True, f"{symbol}: 0 days"
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        return True, f"{symbol}: {len(frame)} days"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: {exc}"


def run_phase(
    name: str,
    fn,
    symbols: list[str],
    workers: int,
    logger: logging.Logger,
    *args,
    symbol_args: dict[str, tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Run a download phase concurrently.

    The function is called with shared args, then symbol and logger. When
    symbol_args is supplied, its per-symbol tuples replace the shared args.
    """
    logger.info(f"=== Phase: {name} ({len(symbols)} symbols, {workers} workers) ===")
    start = time.time()
    results: list[tuple[str, bool, str]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_symbol = {
            pool.submit(fn, *(symbol_args[sym] if symbol_args else args), sym, logger): sym
            for sym in symbols
        }
        for future in cf.as_completed(future_to_symbol):
            sym = future_to_symbol[future]
            try:
                outcome = future.result()
            except Exception as exc:  # noqa: BLE001
                outcome = (False, f"{sym}: {exc}")
            # Normalize: fn may return (ok, msg) or (ok, msg, extra).
            success, msg = bool(outcome[0]), str(outcome[1])
            results.append((sym, success, msg))
            if success:
                logger.info(f"  OK   {msg}")
            else:
                logger.warning(f"  FAIL {msg}")
    elapsed = time.time() - start
    ok = sum(1 for _, success, _ in results if success)
    failed = len(results) - ok
    logger.info(f"Phase {name} done in {elapsed:.1f}s: {ok} ok, {failed} failed")
    return {
        "phase": name,
        "ok": ok,
        "failed": failed,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Polygon market-data database for a named US equity universe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY"),
                        help="Polygon API key (or set POLYGON_API_KEY env var)")
    parser.add_argument("--universe", default="nasdaq100",
                        choices=["nasdaq100", "sp500", "custom"],
                        help="Named universe; also controls the default data-dir")
    parser.add_argument("--universe-file", default=None,
                        help=("CSV universe file. Expected symbol column plus optional listed_date, "
                              "delisted_date, active/status, and exclusion_reason columns."))
    parser.add_argument("--data-dir", default=None,
                        help="Root directory for the downloaded database (default: data/polygon_<universe>)")
    parser.add_argument("--start", default="2024-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.now(UTC).strftime("%Y-%m-%d"),
                        help="End date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--listed-since", default="2024-01-01",
                        help="Include symbols listed at any point on/after this date")
    parser.add_argument("--symbols", default="all",
                        help="Comma-separated tickers, path to CSV, or 'all' (overridden by --universe-file)")
    parser.add_argument("--exclude-symbols", default="",
                        help="Comma-separated tickers or CSV to exclude from the universe")
    parser.add_argument("--respect-source-exclusions", action="store_true",
                        help="Exclude rows that already have source exclusion_reason values")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent download workers (default: 4)")
    parser.add_argument("--base-url", default="https://api.massiveprivateserver.site")
    parser.add_argument("--log-file", default=None, help="Log file path (default: <data-dir>/download.log)")
    # Phase toggles (default: all enabled)
    parser.add_argument("--skip-daily", action="store_true", help="Skip daily-aggregate phase")
    parser.add_argument("--skip-minute", action="store_true", help="Skip minute-aggregate phase")
    parser.add_argument("--skip-options-contracts", action="store_true")
    parser.add_argument("--skip-options-aggregates", action="store_true")
    parser.add_argument("--skip-options-snapshots", action="store_true")
    parser.add_argument("--minute", action="store_true",
                        help="Explicitly enable the minute-aggregate phase (off by default due to size)")
    parser.add_argument("--skip-financials", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--skip-dividends", action="store_true")
    parser.add_argument("--skip-ticker-details", action="store_true")
    parser.add_argument("--skip-open-close", action="store_true")
    parser.add_argument("--open-close", action="store_true",
                        help="Enable the open/close phase (1 request per symbol per day; "
                             "~62000 requests for 97 symbols × 639 days)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: --api-key or POLYGON_API_KEY env var is required", file=sys.stderr)
        return 2

    default_data_dir = Path("data") / f"polygon_{args.universe}"
    data_dir = Path(args.data_dir or default_data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else data_dir / "download.log"
    logger = setup_logging(log_file)
    logger.info(f"Polygon database build started → {data_dir}")

    universe_table = build_universe_table(args)
    universe_path = data_dir / "universe.csv"
    universe_table.to_csv(universe_path, index=False)
    symbols = universe_table.loc[universe_table["included"], "symbol"].astype(str).tolist()
    symbol_windows = build_symbol_windows(universe_table, args.start, args.end)
    symbols = [symbol for symbol in symbols if symbol in symbol_windows]
    excluded_count = len(universe_table) - len(symbols)
    logger.info(
        f"Universe {args.universe}: {len(symbols)} included, {excluded_count} excluded "
        f"({', '.join(symbols[:5])}... )"
    )
    logger.info(f"Universe metadata written → {universe_path}")
    if not symbols:
        logger.error("No symbols to download")
        return 2

    # One source instance per phase (minute source has a different interval).
    daily_source = PolygonMarketDataSource(
        cache_dir=data_dir / "daily",
        api_key=args.api_key,
        base_url=args.base_url,
        interval="1d",
        inter_request_delay=0.15,
        max_retries=8,
        retry_max_delay=120.0,
    )
    minute_source = PolygonMarketDataSource(
        cache_dir=data_dir / "minute",
        api_key=args.api_key,
        base_url=args.base_url,
        interval="1m",
        inter_request_delay=0.15,
        max_retries=8,
        retry_max_delay=120.0,
    )
    options_source = PolygonMarketDataSource(
        cache_dir=data_dir / "options",
        api_key=args.api_key,
        base_url=args.base_url,
        interval="1d",
        inter_request_delay=0.15,
        max_retries=8,
        retry_max_delay=120.0,
    )

    phases: list[dict[str, Any]] = []

    if not args.skip_daily:
        phases.append(run_phase(
            "daily",
            fetch_symbol_daily,
            symbols,
            args.workers,
            logger,
            symbol_args={
                symbol: (daily_source, *symbol_windows[symbol]) for symbol in symbols
            },
        ))

    daily_coverage_windows = build_daily_coverage_windows(
        symbols, symbol_windows, daily_source.cache_dir, logger
    )

    if args.minute and not args.skip_minute:
        phases.append(run_phase(
            "minute",
            fetch_symbol_minute,
            symbols,
            max(1, args.workers // 2),
            logger,
            symbol_args={
                symbol: (minute_source, *daily_coverage_windows[symbol]) for symbol in symbols
            },
        ))

    if not args.skip_options_contracts:
        phases.append(run_phase(
            "options-contracts", fetch_option_contracts, symbols, args.workers, logger, options_source
        ))

    if not args.skip_options_aggregates:
        phases.append(run_phase(
            "options-aggregates",
            fetch_option_aggregates_for_symbol,
            symbols,
            args.workers,
            logger,
            symbol_args={
                symbol: (options_source, *daily_coverage_windows[symbol]) for symbol in symbols
            },
        ))

    if not args.skip_options_snapshots:
        phases.append(run_phase(
            "options-snapshots", fetch_option_snapshots_for_symbol, symbols, args.workers, logger, options_source
        ))

    if not args.skip_financials:
        phases.append(run_phase(
            "financials", fetch_financials, symbols, args.workers, logger, options_source
        ))

    if not args.skip_news:
        phases.append(run_phase(
            "news", fetch_news, symbols, args.workers, logger, options_source
        ))

    if not args.skip_splits:
        phases.append(run_phase(
            "splits", fetch_splits, symbols, args.workers, logger, options_source
        ))

    if not args.skip_dividends:
        phases.append(run_phase(
            "dividends", fetch_dividends, symbols, args.workers, logger, options_source
        ))

    if not args.skip_ticker_details:
        phases.append(run_phase(
            "ticker-details",
            fetch_ticker_details_for_window,
            symbols,
            args.workers,
            logger,
            symbol_args={
                symbol: (options_source, *daily_coverage_windows[symbol]) for symbol in symbols
            },
        ))

    if args.open_close and not args.skip_open_close:
        total_requests = sum(
            len(pd.bdate_range(start=window_start, end=window_end))
            for window_start, window_end in daily_coverage_windows.values()
        )
        logger.warning(
            f"open-close phase: ~{total_requests} lifecycle-adjusted requests "
            f"across {len(symbols)} symbols — 1 request per symbol per business day"
        )
        phases.append(run_phase(
            "open-close",
            fetch_open_close,
            symbols,
            max(1, args.workers // 2),
            logger,
            symbol_args={
                symbol: (options_source, *daily_coverage_windows[symbol]) for symbol in symbols
            },
        ))

    # Final manifest.
    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "universe": args.universe,
        "universe_file": args.universe_file,
        "universe_metadata": str(universe_path),
        "listed_since": args.listed_since,
        "start": args.start,
        "end": args.end,
        "symbols": symbols,
        "symbol_windows": {
            symbol: {"start": window[0], "end": window[1]}
            for symbol, window in symbol_windows.items()
        },
        "daily_coverage_windows": {
            symbol: {"start": window[0], "end": window[1]}
            for symbol, window in daily_coverage_windows.items()
        },
        "excluded_symbols": universe_table.loc[
            ~universe_table["included"], ["symbol", "exclusion_reason"]
        ].to_dict(orient="records"),
        "phases": [
            {"phase": p["phase"], "ok": p["ok"], "failed": p["failed"], "elapsed_s": p["elapsed_s"]}
            for p in phases
        ],
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest written → {manifest_path}")

    # Summary.
    total_ok = sum(p["ok"] for p in phases)
    total_failed = sum(p["failed"] for p in phases)
    logger.info(f"Build complete: {total_ok} ok, {total_failed} failed across {len(phases)} phases")

    # Print directory tree summary.
    try:
        total_bytes = 0
        for path in data_dir.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size
        logger.info(f"Total database size: {total_bytes / 1e6:.1f} MB")
    except Exception:  # noqa: BLE001
        pass

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1) from None
