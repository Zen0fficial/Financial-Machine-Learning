from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from us_factor_screening.data import (
    MarketDataError,
    MarketDataPanel,
    MarketDataProvenance,
    MarketDataResult,
)
from us_factor_screening.providers import (
    AlpacaMarketDataSource,
    FrozenMarketDataSource,
    HttpResponse,
)
from us_factor_screening.reporting import write_data_outputs


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1_000_000, 1_100_000],
        }
    )


def test_market_data_panel_rejects_mixed_provider_definitions() -> None:
    now = datetime(2024, 1, 3, tzinfo=UTC)
    first = MarketDataResult(
        "EWY",
        _frame(),
        MarketDataProvenance("yahoo_finance", "yahoo_chart", "all"),
        now,
    )
    second = MarketDataResult(
        "SPY",
        _frame(),
        MarketDataProvenance("alpaca", "iex", "all"),
        now,
    )

    with pytest.raises(MarketDataError, match="mixed market-data"):
        MarketDataPanel.from_results([first, second])


def test_alpaca_source_uses_explicit_feed_adjustment_and_cache(tmp_path: Path) -> None:
    urls = []
    payload = {
        "bars": [
            {"t": "2024-01-02T05:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 10},
            {"t": "2024-01-03T05:00:00Z", "o": 101, "h": 102, "l": 100, "c": 101.5, "v": 11},
        ],
        "next_page_token": None,
    }

    def transport(request, timeout):
        urls.append(request.full_url)
        return HttpResponse(
            200,
            json.dumps(payload).encode(),
            {"X-Request-ID": "request-1"},
        )

    source = AlpacaMarketDataSource(
        tmp_path,
        api_key="key",
        secret_key="secret",
        feed="iex",
        adjustment="all",
        transport=transport,
    )
    first = source.fetch_result("ewy", "2024-01-02", "2024-01-03")
    second = source.fetch_result("EWY", "2024-01-02", "2024-01-03")

    query = parse_qs(urlparse(urls[0]).query)
    assert query["feed"] == ["iex"]
    assert query["adjustment"] == ["all"]
    assert first.provenance == source.provenance
    assert first.metadata["request_ids"] == ["request-1"]
    assert not first.cache_hit
    assert second.cache_hit
    assert len(urls) == 1


def test_frozen_csv_round_trip_preserves_manifest_provenance(tmp_path: Path) -> None:
    provenance = MarketDataProvenance("yahoo_finance", "yahoo_chart", "all")
    now = datetime(2024, 1, 3, tzinfo=UTC)
    panel = MarketDataPanel.from_results(
        [
            MarketDataResult("EWY", _frame(), provenance, now),
            MarketDataResult("SPY", _frame(), provenance, now),
        ]
    )
    quality = pd.DataFrame({"symbol": ["EWY", "SPY"], "ok": [True, True]})
    write_data_outputs(panel, quality, tmp_path)

    frozen = FrozenMarketDataSource(tmp_path)
    replay = frozen.fetch_many(["EWY", "SPY"], "2024-01-02", "2024-01-03")

    assert replay.provenance == provenance
    pd.testing.assert_frame_equal(replay["EWY"], _frame())


def test_frozen_source_rejects_snapshot_changed_after_manifest(tmp_path: Path) -> None:
    provenance = MarketDataProvenance("yahoo_finance", "yahoo_chart", "all")
    result = MarketDataResult(
        "EWY",
        _frame(),
        provenance,
        datetime(2024, 1, 3, tzinfo=UTC),
    )
    panel = MarketDataPanel.from_results([result])
    write_data_outputs(panel, pd.DataFrame({"symbol": ["EWY"]}), tmp_path)
    with (tmp_path / "ohlcv.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    source = FrozenMarketDataSource(tmp_path)
    with pytest.raises(MarketDataError, match="checksum"):
        source.fetch("EWY", "2024-01-02", "2024-01-03")


def test_frozen_source_rejects_embedded_mixed_feed(tmp_path: Path) -> None:
    data = pd.concat(
        [
            _frame().assign(Symbol="EWY", Feed="iex"),
            _frame().assign(Symbol="SPY", Feed="sip"),
        ],
        ignore_index=True,
    )
    source_path = tmp_path / "mixed.csv"
    data.to_csv(source_path, index=False)
    provenance = MarketDataProvenance("alpaca", "iex", "all")

    source = FrozenMarketDataSource(source_path, provenance=provenance)
    with pytest.raises(MarketDataError, match="mixes feed"):
        source.fetch("EWY", "2024-01-02", "2024-01-03")
