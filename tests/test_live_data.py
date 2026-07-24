from __future__ import annotations

import os

import pytest

from us_factor_screening.data import YahooFinanceSource, validate_ohlcv


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_MARKET_TESTS") != "1",
    reason="set RUN_LIVE_MARKET_TESTS=1 to access Yahoo Finance",
)
def test_live_ewy_data() -> None:
    source = YahooFinanceSource()
    frame = source.fetch("EWY", "2025-01-02", "2025-03-31", refresh=True)
    report = validate_ohlcv(frame, "EWY")
    assert report.ok
    assert report.rows >= 50
