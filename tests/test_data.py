from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from us_factor_screening.data import (
    MarketDataError,
    MarketDataPanel,
    YahooFinanceSource,
    normalize_ohlcv,
    validate_ohlcv,
    validate_us_symbol,
)


def _response() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [60.0, 60.5],
            "High": [61.0, 62.0],
            "Low": [59.5, 60.25],
            "Close": [60.5, 61.75],
            "Volume": [1_000_000, 1_100_000],
        }
    )


def test_yahoo_source_downloads_inclusive_range_and_caches(tmp_path: Path) -> None:
    calls = []

    def fetcher(symbol: str, start: str, end: str) -> pd.DataFrame:
        calls.append((symbol, start, end))
        return _response()

    source = YahooFinanceSource(cache_dir=tmp_path, fetcher=fetcher)
    first = source.fetch("ewy", "2024-01-02", "2024-01-03")
    second = source.fetch("EWY", "2024-01-02", "2024-01-03")

    assert calls == [("EWY", "2024-01-02", "2024-01-04")]
    pd.testing.assert_frame_equal(first, second)
    assert validate_ohlcv(first, "EWY").ok


def test_yahoo_fetch_many_returns_uniform_backward_total_return_panel(tmp_path: Path) -> None:
    source = YahooFinanceSource(cache_dir=tmp_path, fetcher=lambda *_: _response())

    panel = source.fetch_many(["EWY", "SPY"], "2024-01-02", "2024-01-03")

    assert isinstance(panel, MarketDataPanel)
    assert panel.provenance.provider == "yahoo_finance"
    assert panel.provenance.adjustment == "backward_total_return"
    assert set(panel) == {"EWY", "SPY"}
    assert panel.results["EWY"].metadata["corporate_actions"] == ("splits_and_cash_distributions")


def test_yahoo_source_rejects_stale_response(tmp_path: Path) -> None:
    source = YahooFinanceSource(cache_dir=tmp_path, fetcher=lambda *_: _response())

    with pytest.raises(MarketDataError, match="stale"):
        source.fetch("EWY", "2024-01-02", "2024-03-31")


def test_yahoo_source_retries_rate_limits(tmp_path: Path) -> None:
    attempts = 0
    delays = []

    def fetcher(*_: str) -> pd.DataFrame:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise YFRateLimitError()
        return _response()

    source = YahooFinanceSource(
        cache_dir=tmp_path,
        fetcher=fetcher,
        max_retries=2,
        retry_base_delay=0.5,
        inter_request_delay=0.0,
        jitter=0.0,
        sleeper=delays.append,
    )

    assert len(source.fetch("EWY", "2024-01-02", "2024-01-03")) == 2
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_yahoo_source_429_uses_longer_exponential_backoff(tmp_path: Path) -> None:
    """429 retries fall back to the longer rate-limit base with exponential growth."""
    attempts = 0
    delays: list[float] = []

    def fetcher(*_: str) -> pd.DataFrame:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise YFRateLimitError()
        return _response()

    # rng returns 0.5 -> jitter factor = 1 + (0.5*2-1)*0.25 = 1.0 (deterministic).
    source = YahooFinanceSource(
        cache_dir=tmp_path,
        fetcher=fetcher,
        max_retries=3,
        retry_base_delay=30.0,
        inter_request_delay=0.0,
        jitter=0.25,
        sleeper=delays.append,
        rng=lambda: 0.5,
    )

    assert len(source.fetch("EWY", "2024-01-02", "2024-01-03")) == 2
    assert attempts == 3
    # 30s, 60s — no jitter because rng centers on 0.5.
    assert delays == [30.0, 60.0]


def test_yahoo_source_429_honors_retry_after_resolver(tmp_path: Path) -> None:
    """A Retry-After resolver overrides the exponential backoff when present."""
    attempts = 0
    delays: list[float] = []
    resolver_calls: list[BaseException] = []

    def fetcher(*_: str) -> pd.DataFrame:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise YFRateLimitError()
        return _response()

    def resolver(exc: BaseException) -> float | None:
        resolver_calls.append(exc)
        return 42.0

    source = YahooFinanceSource(
        cache_dir=tmp_path,
        fetcher=fetcher,
        max_retries=2,
        retry_base_delay=30.0,
        inter_request_delay=2.0,
        jitter=0.0,
        retry_after_resolver=resolver,
        sleeper=delays.append,
    )

    assert len(source.fetch("EWY", "2024-01-02", "2024-01-03")) == 2
    assert attempts == 2
    assert len(resolver_calls) == 1
    assert isinstance(resolver_calls[0], YFRateLimitError)
    # Retry-After (42s) wins on the 429, then 2s inter-request throttle on success.
    assert delays == [42.0, 2.0]


def test_yahoo_source_throttles_between_successful_symbol_fetches(tmp_path: Path) -> None:
    """fetch_many inserts an inter-request delay after each successful download."""
    delays: list[float] = []

    def fetcher(*_: str) -> pd.DataFrame:
        return _response()

    source = YahooFinanceSource(
        cache_dir=tmp_path,
        fetcher=fetcher,
        inter_request_delay=1.0,
        jitter=0.0,
        sleeper=delays.append,
    )

    panel = source.fetch_many(["EWY", "SPY"], "2024-01-02", "2024-01-03")
    assert set(panel) == {"EWY", "SPY"}
    # One inter-request delay per successful fetch (including the last one —
    # the throttle is applied whenever the network is touched, which is what
    # keeps a follow-up call from hammering Yahoo immediately).
    assert delays == [1.0, 1.0]


def test_normalize_rejects_missing_volume() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
        }
    )
    with pytest.raises(MarketDataError, match="Volume"):
        normalize_ohlcv(frame, "EWY")


def test_quality_report_catches_impossible_bar() -> None:
    frame = normalize_ohlcv(
        pd.DataFrame(
            {
                "Date": ["2024-01-02"],
                "Open": [60],
                "High": [59],
                "Low": [58],
                "Close": [61],
                "Volume": [100],
            }
        ),
        "EWY",
    )
    report = validate_ohlcv(frame, "EWY")
    assert not report.ok
    assert report.invalid_ohlc_rows == 1


def test_quality_report_catches_duplicate_date() -> None:
    frame = normalize_ohlcv(
        pd.DataFrame(
            {
                "Date": ["2024-01-02", "2024-01-02"],
                "Open": [60, 60],
                "High": [61, 61],
                "Low": [59, 59],
                "Close": [60.5, 60.5],
                "Volume": [100, 100],
            }
        ),
        "EWY",
    )
    assert validate_ohlcv(frame, "EWY").duplicate_dates == 1


def test_us_symbol_scope() -> None:
    assert validate_us_symbol("ewy") == "EWY"
    with pytest.raises(ValueError, match="Non-US"):
        validate_us_symbol("0700.HK")
