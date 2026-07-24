"""Tests for the Polygon shared-API market-data provider.

All tests use injected ``transport`` and ``sleeper`` callables so no real
HTTP calls are made.  The single live smoke test at the bottom is gated
behind ``RUN_LIVE_MARKET_TESTS``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from us_factor_screening.data import MarketDataError
from us_factor_screening.providers import HttpResponse, PolygonMarketDataSource

# 2024-01-02 00:00:00 UTC and 2024-01-03 00:00:00 UTC in milliseconds.
_T_2024_01_02 = 1_704_153_600_000
_T_2024_01_03 = 1_704_240_000_000


def _daily_bars() -> list[dict]:
    return [
        {
            "v": 1_000_000, "vw": 100.5, "o": 100.0, "c": 100.5,
            "h": 101.0, "l": 99.0, "t": _T_2024_01_02, "n": 5000,
        },
        {
            "v": 1_100_000, "vw": 101.2, "o": 101.0, "c": 101.5,
            "h": 102.0, "l": 100.0, "t": _T_2024_01_03, "n": 5500,
        },
    ]


def _ok_response(results: list, next_url: str | None = None) -> HttpResponse:
    payload: dict = {"results": results}
    if next_url:
        payload["next_url"] = next_url
    return HttpResponse(200, json.dumps(payload).encode(), {})


class FakeTransport:
    """Queued transport that records every request and returns canned responses."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = list(responses)
        self.requests: list = []

    def __call__(self, request, timeout: float) -> HttpResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("FakeTransport: no queued response remaining")
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# 1. Construction and validation
# ---------------------------------------------------------------------------

def test_construction_rejects_bad_interval() -> None:
    with pytest.raises(ValueError, match="interval"):
        PolygonMarketDataSource(api_key="k", interval="2d")


def test_construction_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(MarketDataError, match="api_key"):
        PolygonMarketDataSource()


def test_construction_reads_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    src = PolygonMarketDataSource()
    assert src.api_key == "env-key"


def test_provenance_defaults() -> None:
    src = PolygonMarketDataSource(api_key="k")
    prov = src.provenance
    assert prov.provider == "polygon"
    assert prov.feed == "aggregates"
    assert prov.adjustment == "backward_total_return"
    assert prov.interval == "1d"
    assert prov.session == "regular"


def test_provenance_unadjusted_minute() -> None:
    src = PolygonMarketDataSource(api_key="k", interval="1m", adjusted=False)
    assert src.provenance.adjustment == "raw"
    assert src.provenance.interval == "1m"


# ---------------------------------------------------------------------------
# 2. URL construction
# ---------------------------------------------------------------------------

def test_url_daily() -> None:
    src = PolygonMarketDataSource(api_key="SECRET", interval="1d")
    url = src._build_url("AAPL", "2024-01-02", "2024-01-03")
    parsed = urlparse(url)
    assert parsed.path == "/v2/aggs/ticker/AAPL/range/1/day/2024-01-02/2024-01-03"
    qs = parse_qs(parsed.query)
    assert qs["adjusted"] == ["true"]
    assert qs["sort"] == ["asc"]
    assert qs["limit"] == ["50000"]
    assert qs["apiKey"] == ["SECRET"]


def test_url_minute() -> None:
    src = PolygonMarketDataSource(api_key="k", interval="1m")
    assert "/range/1/minute/" in src._build_url("AAPL", "2024-01-02", "2024-01-03")


def test_url_5minute() -> None:
    src = PolygonMarketDataSource(api_key="k", interval="5m")
    assert "/range/5/minute/" in src._build_url("AAPL", "2024-01-02", "2024-01-03")


def test_url_unadjusted() -> None:
    src = PolygonMarketDataSource(api_key="k", adjusted=False)
    assert "adjusted=false" in src._build_url("AAPL", "2024-01-02", "2024-01-03")


def test_append_api_key() -> None:
    src = PolygonMarketDataSource(api_key="KEY")
    assert src._append_api_key("https://x.io/p?a=1") == "https://x.io/p?a=1&apiKey=KEY"
    assert src._append_api_key("https://x.io/p") == "https://x.io/p?apiKey=KEY"
    assert src._append_api_key("https://x.io/p?apiKey=OLD") == "https://x.io/p?apiKey=OLD"


# ---------------------------------------------------------------------------
# 3. Response parsing
# ---------------------------------------------------------------------------

def test_parse_results_maps_columns_and_timestamps() -> None:
    src = PolygonMarketDataSource(api_key="k", interval="1d")
    frame = src._parse_results(_daily_bars())
    assert "Date" in frame.columns
    assert "Open" in frame.columns
    assert "High" in frame.columns
    assert "Low" in frame.columns
    assert "Close" in frame.columns
    assert "Volume" in frame.columns
    assert "VWAP" in frame.columns
    assert "TradeCount" in frame.columns
    assert frame["Open"].tolist() == [100.0, 101.0]
    assert frame["Close"].tolist() == [100.5, 101.5]
    assert frame["VWAP"].tolist() == [100.5, 101.2]
    # Daily bars must be normalised to midnight.
    assert frame["Date"].iloc[0] == pd.Timestamp("2024-01-02")
    assert frame["Date"].iloc[1] == pd.Timestamp("2024-01-03")


def test_parse_results_minute_keeps_time() -> None:
    src = PolygonMarketDataSource(api_key="k", interval="1m")
    bar = dict(_daily_bars()[0])
    bar["t"] = _T_2024_01_02 + 13 * 3_600_000 + 30 * 60_000  # 13:30 UTC
    frame = src._parse_results([bar])
    assert frame["Date"].iloc[0] == pd.Timestamp("2024-01-02 13:30:00")


def test_parse_results_empty_raises() -> None:
    src = PolygonMarketDataSource(api_key="k")
    with pytest.raises(MarketDataError, match="no bars"):
        src._parse_results([])


# ---------------------------------------------------------------------------
# 4. Pagination
# ---------------------------------------------------------------------------

def test_pagination_follows_next_url() -> None:
    next_url = (
        "https://api.example.com/v2/aggs/ticker/AAPL/range/1/day/"
        "2024-01-02/2024-01-03?cursor=abc"
    )
    transport = FakeTransport(
        [
            _ok_response(_daily_bars()[:1], next_url=next_url),
            _ok_response(_daily_bars()[1:]),
        ]
    )
    src = PolygonMarketDataSource(
        api_key="k", transport=transport, sleeper=lambda _: None,
        inter_request_delay=0.0,
    )
    frame = src._download("AAPL", "2024-01-02", "2024-01-03")
    assert len(transport.requests) == 2
    # The second request URL must have the API key appended.
    second_url = transport.requests[1].full_url
    assert "apiKey=k" in second_url
    assert "cursor=abc" in second_url
    assert len(frame) == 2


# ---------------------------------------------------------------------------
# 5. Rate-limit retry
# ---------------------------------------------------------------------------

def test_rate_limit_retry_then_success() -> None:
    transport = FakeTransport(
        [
            HttpResponse(429, b"rate limited", {"Retry-After": "1"}),
            _ok_response(_daily_bars()),
        ]
    )
    sleeps: list[float] = []
    src = PolygonMarketDataSource(
        api_key="k", transport=transport, sleeper=lambda d: sleeps.append(d),
        inter_request_delay=0.0,
    )
    frame = src._download("AAPL", "2024-01-02", "2024-01-03")
    assert len(transport.requests) == 2
    assert sleeps == [1.0]
    assert len(frame) == 2


def test_rate_limit_exhausts_retries() -> None:
    transport = FakeTransport(
        [HttpResponse(429, b"rate limited", {})] * 10
    )
    sleeps: list[float] = []
    src = PolygonMarketDataSource(
        api_key="k", transport=transport, sleeper=lambda d: sleeps.append(d),
        max_retries=2, retry_base_delay=0.01, inter_request_delay=0.0,
    )
    with pytest.raises(MarketDataError, match="rate limit"):
        src._download("AAPL", "2024-01-02", "2024-01-03")
    # 2 retries => 3 total attempts => 2 sleeps.
    assert len(sleeps) == 2


def test_auth_error_raised() -> None:
    transport = FakeTransport([HttpResponse(401, b"unauthorized", {})])
    src = PolygonMarketDataSource(
        api_key="bad", transport=transport, sleeper=lambda _: None,
    )
    with pytest.raises(MarketDataError, match="authentication"):
        src._download("AAPL", "2024-01-02", "2024-01-03")


def test_not_found_error_raised() -> None:
    transport = FakeTransport([HttpResponse(404, b'{"detail":"not found"}', {})])
    src = PolygonMarketDataSource(
        api_key="k", transport=transport, sleeper=lambda _: None,
    )
    with pytest.raises(MarketDataError, match="404"):
        src._download("AAPL", "2024-01-02", "2024-01-03")


# ---------------------------------------------------------------------------
# 6. Caching
# ---------------------------------------------------------------------------

def test_caching_creates_file_and_hits_on_second_call(tmp_path: Path) -> None:
    transport = FakeTransport([_ok_response(_daily_bars())])
    src = PolygonMarketDataSource(
        api_key="k", cache_dir=tmp_path, transport=transport,
        sleeper=lambda _: None, inter_request_delay=0.0,
    )
    result1 = src.fetch_result("AAPL", "2024-01-02", "2024-01-03")
    assert not result1.cache_hit
    assert result1.provenance.provider == "polygon"
    assert "VWAP" in result1.frame.columns
    assert "TradeCount" in result1.frame.columns
    assert result1.frame["VWAP"].iloc[0] == 100.5

    cache_file = tmp_path / "polygon_1d_backward_total_return_AAPL_20240102_20240103.csv"
    assert cache_file.exists()

    result2 = src.fetch_result("AAPL", "2024-01-02", "2024-01-03")
    assert result2.cache_hit
    assert len(transport.requests) == 1
    pd.testing.assert_frame_equal(result1.frame, result2.frame, check_dtype=False)


def test_minute_data_uses_gz_cache(tmp_path: Path) -> None:
    bar = dict(_daily_bars()[0])
    bar["t"] = _T_2024_01_02 + 13 * 3_600_000  # 13:00 UTC
    transport = FakeTransport([_ok_response([bar])])
    src = PolygonMarketDataSource(
        api_key="k", cache_dir=tmp_path, interval="1m", transport=transport,
        sleeper=lambda _: None, inter_request_delay=0.0,
    )
    src.fetch_result("AAPL", "2024-01-02", "2024-01-03")
    cache_path = src._cache_path("AAPL", "2024-01-02", "2024-01-03")
    assert cache_path.suffix == ".gz"
    assert cache_path.exists()
    # Verify the gzipped cache is readable.
    reloaded = pd.read_csv(cache_path)
    assert len(reloaded) == 1


# ---------------------------------------------------------------------------
# 7. Option contract fetching
# ---------------------------------------------------------------------------

def test_fetch_option_contracts(tmp_path: Path) -> None:
    contracts = [
        {
            "cfi": "OCASPS", "contract_type": "call", "exercise_style": "american",
            "expiration_date": "2026-08-21", "primary_exchange": "BATO",
            "shares_per_contract": 100, "strike_price": 225.0,
            "ticker": "O:AAPL260821C00225000", "underlying_ticker": "AAPL",
        },
        {
            "cfi": "OCASPS", "contract_type": "put", "exercise_style": "american",
            "expiration_date": "2026-08-21", "primary_exchange": "BATO",
            "shares_per_contract": 100, "strike_price": 225.0,
            "ticker": "O:AAPL260821P00225000", "underlying_ticker": "AAPL",
        },
    ]
    transport = FakeTransport([_ok_response(contracts)])
    src = PolygonMarketDataSource(
        api_key="k", cache_dir=tmp_path, transport=transport,
        sleeper=lambda _: None, inter_request_delay=0.0,
    )
    frame = src.fetch_option_contracts("AAPL")
    assert len(frame) == 2
    assert "ticker" in frame.columns
    assert "contract_type" in frame.columns
    assert "strike_price" in frame.columns
    assert frame["ticker"].iloc[0] == "O:AAPL260821C00225000"
    assert frame["contract_type"].tolist() == ["call", "put"]
    # Cache file should exist.
    assert (tmp_path / "options_contracts_AAPL.csv").exists()


# ---------------------------------------------------------------------------
# 8. Option snapshot parsing
# ---------------------------------------------------------------------------

def test_fetch_option_snapshot(tmp_path: Path) -> None:
    snapshot = {
        "results": {
            "greeks": {"delta": 0.987, "gamma": 0.0006, "theta": -0.050, "vega": 0.037},
            "implied_volatility": 0.601,
            "open_interest": 629,
            "last_quote": {"bid": 95.4, "ask": 98.45},
            "day": {"close": 103.15, "volume": 1},
            "details": {
                "contract_type": "call",
                "expiration_date": "2026-08-21",
                "strike_price": 225,
            },
            "underlying_asset": {"price": 321.1},
        }
    }
    transport = FakeTransport(
        [HttpResponse(200, json.dumps(snapshot).encode(), {})]
    )
    src = PolygonMarketDataSource(
        api_key="k", cache_dir=tmp_path, transport=transport,
        sleeper=lambda _: None,
    )
    result = src.fetch_option_snapshot("AAPL", "O:AAPL260821C00225000")
    assert result["greeks"]["delta"] == 0.987
    assert result["implied_volatility"] == 0.601
    assert result["open_interest"] == 629
    assert result["details"]["strike_price"] == 225


def test_fetch_option_aggregates(tmp_path: Path) -> None:
    transport = FakeTransport([_ok_response(_daily_bars())])
    src = PolygonMarketDataSource(
        api_key="k", cache_dir=tmp_path, transport=transport,
        sleeper=lambda _: None, inter_request_delay=0.0,
    )
    frame = src.fetch_option_aggregates(
        "O:AAPL260821C00225000", "2024-01-02", "2024-01-03"
    )
    assert len(frame) == 2
    assert "Close" in frame.columns
    assert frame["Close"].tolist() == [100.5, 101.5]
    # Verify the option contract ticker is sanitised in the cache filename.
    cache_files = list(tmp_path.glob("polygon_option_O_AAPL260821C00225000_*.csv"))
    assert len(cache_files) == 1


# ---------------------------------------------------------------------------
# 9. Live smoke test (gated)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_MARKET_TESTS"),
    reason="set RUN_LIVE_MARKET_TESTS=1 to access the Polygon API",
)
def test_live_polygon_aapl_daily() -> None:
    src = PolygonMarketDataSource()  # reads POLYGON_API_KEY from env
    result = src.fetch_result("AAPL", "2026-07-15", "2026-07-22", refresh=True)
    assert len(result.frame) >= 3
    assert result.provenance.provider == "polygon"
    assert result.provenance.feed == "aggregates"
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(result.frame.columns)
