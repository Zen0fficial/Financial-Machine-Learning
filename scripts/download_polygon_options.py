"""Download Polygon option contract reference for all Nasdaq-100 symbols.

Free tier (Options Basic) provides:
- Contract reference (strikes, expirations, type, exchange) — historical + active
- Previous-day OHLCV per contract — current only, NOT historical

This script downloads the contract reference for all eligible Nasdaq-100
symbols with expiration dates >= 2024-01-01. The result is a single CSV
mapping every option contract that existed during the research window.

Rate limit: 5 calls/min on the free tier. The script throttles to 4 calls/min
with a safety margin.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None
BASE = "https://api.polygon.io"
KEY = os.environ.get("POLYGON_API_KEY", "")

# Throttle: 4 calls/min to stay under the 5/min limit
MIN_INTERVAL = 16.0  # seconds between calls (≈3.75/min)


def fetch_contracts(symbol: str, expiration_gte: str = "2024-01-01") -> list[dict]:
    """Fetch all option contracts for a symbol with expiration >= expiration_gte."""
    results: list[dict] = []
    url = f"{BASE}/v3/reference/options/contracts"
    params: dict = {
        "apiKey": KEY,
        "underlying_ticker": symbol,
        "limit": 1000,
        "expired": "true",
        "expiration_date.gte": expiration_gte,
        "sort": "expiration_date",
        "order": "asc",
    }

    page = 0
    while url:
        t0 = time.time()
        try:
            r = requests.get(url, params=params, proxies=PROXIES, timeout=30)
        except requests.exceptions.ProxyError as e:
            print(f"    proxy error, retrying in 30s: {e}")
            time.sleep(30)
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"    connection error, retrying in 30s: {e}")
            time.sleep(30)
            continue
        elapsed = time.time() - t0

        if r.status_code == 429:
            print("    429 rate limited, sleeping 60s...")
            time.sleep(60)
            continue
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {r.text[:150]}")
            break

        d = r.json()
        batch = d.get("results", [])
        results.extend(batch)
        page += 1
        print(f"    page {page}: +{len(batch)} contracts (total {len(results)})")

        # Follow pagination
        next_url = d.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": KEY}  # next_url already has query params
        else:
            url = None

        # Throttle
        wait = MIN_INTERVAL - elapsed
        if wait > 0 and url:
            time.sleep(wait)

    return results


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    uni_path = project / "data" / "free_nasdaq100_2024_2026" / "nasdaq100_universe.csv"
    out_dir = project / "data" / "polygon_options"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "option_contracts.csv"

    uni = pd.read_csv(uni_path)
    symbols = uni[uni["eligible"] == True]["symbol"].tolist()  # noqa: E712
    print(f"Downloading option contracts for {len(symbols)} symbols...")
    print("Rate limit: ~4 calls/min (free tier allows 5/min)")
    print(f"Output: {out_csv}")
    print()

    all_contracts: list[dict] = []
    t_start = time.time()

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}...")
        contracts = fetch_contracts(sym)
        all_contracts.extend(contracts)

        elapsed = time.time() - t_start
        rate = len(all_contracts) / max(elapsed, 1)
        print(f"  → {len(contracts)} contracts | cumulative {len(all_contracts)} | "
              f"{elapsed/60:.1f}min elapsed | {rate:.0f} contracts/s")

        # Checkpoint every 10 symbols
        if i % 10 == 0:
            df = pd.DataFrame(all_contracts)
            df.to_csv(out_csv, index=False)
            print(f"  [checkpoint] saved {len(df)} contracts to {out_csv.name}")

    # Final save
    df = pd.DataFrame(all_contracts)
    df.to_csv(out_csv, index=False)

    total_min = (time.time() - t_start) / 60
    print(f"\nDone in {total_min:.1f}min")
    print(f"Total contracts: {len(df):,}")
    print(f"Symbols: {df['underlying_ticker'].nunique()}")
    print(f"Date range: {df['expiration_date'].min()} to {df['expiration_date'].max()}")
    print(f"Output: {out_csv} ({out_csv.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
