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

# Fallback universe used only if no bundled CSV is found. The bundled
# data/free_nasdaq100_2024_2026/nasdaq100_universe.csv is preferred and
# normally supplies the full 97-symbol eligible universe.
DEFAULT_SYMBOLS = sorted({
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMGN", "AMZN", "ANSS",
})


def load_symbols(args: argparse.Namespace) -> list[str]:
    """Resolve the symbol universe from CLI args or a bundled file."""
    if args.symbols and args.symbols.lower() != "all":
        if Path(args.symbols).exists():
            frame = pd.read_csv(args.symbols)
            # Look for a Symbol/Symbol column; fall back to first column.
            for col in ("Symbol", "symbol", "Ticker", "ticker"):
                if col in frame.columns:
                    return sorted({str(s).strip().upper() for s in frame[col] if str(s).strip()})
            return sorted({str(s).strip().upper() for s in frame.iloc[:, 0] if str(s).strip()})
        return sorted({s.strip().upper() for s in args.symbols.split(",") if s.strip()})
    # Try the existing bundle's universe file.
    for candidate in (
        _PROJECT_ROOT / "data" / "free_nasdaq100_2024_2026" / "nasdaq100_universe.csv",
        _PROJECT_ROOT / "data" / "nasdaq100_universe.csv",
    ):
        if candidate.exists():
            frame = pd.read_csv(candidate)
            for col in ("Symbol", "symbol", "Ticker", "ticker"):
                if col in frame.columns:
                    eligible = frame
                    if "exclusion_reason" in frame.columns:
                        eligible = frame[frame["exclusion_reason"].isna()]
                    return sorted({str(s).strip().upper() for s in eligible[col] if str(s).strip()})
    return DEFAULT_SYMBOLS


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
    source: PolygonMarketDataSource, symbol: str, start: str, end: str, logger: logging.Logger
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


# Lock guarding concurrent appends to the shared ticker_details.csv.
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
    source: PolygonMarketDataSource, symbol: str, logger: logging.Logger
) -> tuple[bool, str]:
    """Download company info for one symbol and append to a combined CSV."""
    try:
        out_path = source.cache_dir / "ticker_details.csv"
        # Best-effort cache check (race conditions are benign — worst case is a duplicate row).
        if out_path.exists():
            try:
                existing = pd.read_csv(out_path)
                if "ticker" in existing.columns and symbol in existing["ticker"].astype(str).values:
                    match = existing.loc[existing["ticker"].astype(str) == symbol]
                    if not match.empty and "name" in match.columns:
                        name = str(match["name"].iloc[0])
                        return True, f"{symbol}: {name} (cached)"
            except Exception:  # noqa: BLE001
                pass
        url = f"{source.base_url}/v3/reference/tickers/{symbol}?apiKey={source.api_key}"
        payload = _fetch_single(source, url)
        results = payload.get("results")
        if not isinstance(results, dict):
            return False, f"{symbol}: no results"
        name = results.get("name") or symbol
        row = pd.json_normalize([results])
        with _TICKER_DETAILS_LOCK:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not out_path.exists()
            row.to_csv(out_path, mode="a", header=write_header, index=False)
        return True, f"{symbol}: {name}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{symbol}: {exc}"


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
) -> dict[str, Any]:
    """Run a download phase concurrently.

    `fn` is called as `fn(*args, symbol, logger)` and returns a tuple whose
    first element is a success bool and whose second is a message string.
    """
    logger.info(f"=== Phase: {name} ({len(symbols)} symbols, {workers} workers) ===")
    start = time.time()
    results: list[tuple[str, bool, str]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_symbol = {pool.submit(fn, *args, sym, logger): sym for sym in symbols}
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
        description="Download a Polygon market-data database for the Nasdaq-100.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY"),
                        help="Polygon API key (or set POLYGON_API_KEY env var)")
    parser.add_argument("--data-dir", default="data/polygon_db",
                        help="Root directory for the downloaded database")
    parser.add_argument("--start", default="2024-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.now(UTC).strftime("%Y-%m-%d"),
                        help="End date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--symbols", default="all",
                        help="Comma-separated tickers, path to CSV, or 'all'")
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

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else data_dir / "download.log"
    logger = setup_logging(log_file)
    logger.info(f"Polygon database build started → {data_dir}")

    symbols = load_symbols(args)
    logger.info(f"Universe: {len(symbols)} symbols ({', '.join(symbols[:5])}... )")
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
            "daily", fetch_symbol_daily, symbols, args.workers, logger, daily_source, args.start, args.end
        ))

    if args.minute and not args.skip_minute:
        phases.append(run_phase(
            "minute", fetch_symbol_minute, symbols, max(1, args.workers // 2), logger, minute_source, args.start, args.end
        ))

    if not args.skip_options_contracts:
        phases.append(run_phase(
            "options-contracts", fetch_option_contracts, symbols, args.workers, logger, options_source
        ))

    if not args.skip_options_aggregates:
        phases.append(run_phase(
            "options-aggregates", fetch_option_aggregates_for_symbol, symbols, args.workers, logger, options_source, args.start, args.end
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
            "ticker-details", fetch_ticker_details, symbols, args.workers, logger, options_source
        ))

    if args.open_close and not args.skip_open_close:
        n_days = len(pd.bdate_range(start=args.start, end=args.end))
        total_requests = n_days * len(symbols)
        logger.warning(
            f"open-close phase: ~{total_requests} requests "
            f"({n_days} trading days × {len(symbols)} symbols) — 1 request per symbol per day"
        )
        phases.append(run_phase(
            "open-close", fetch_open_close, symbols, max(1, args.workers // 2),
            logger, options_source, args.start, args.end
        ))

    # Final manifest.
    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "start": args.start,
        "end": args.end,
        "symbols": symbols,
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
