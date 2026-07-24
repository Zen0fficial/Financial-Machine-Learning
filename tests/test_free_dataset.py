from __future__ import annotations

import json
import tarfile
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import pandas as pd

from us_factor_screening.data import (
    MarketDataPanel,
    MarketDataProvenance,
    MarketDataResult,
)
from us_factor_screening.free_dataset import (
    CBOE_EXCHANGES,
    CboeOptionVolumeSource,
    aggregate_cboe_option_volume,
    fetch_nasdaq100_snapshot,
    select_price_history_eligible,
    write_free_dataset_bundle,
)
from us_factor_screening.providers import HttpResponse


def _response(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, body=body.encode(), headers={})


def _price_result(symbol: str, dates: pd.DatetimeIndex) -> MarketDataResult:
    values = pd.Series(range(100, 100 + len(dates)), dtype="float64")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": values,
            "High": values + 1,
            "Low": values - 1,
            "Close": values,
            "Volume": 1_000,
        }
    )
    return MarketDataResult(
        symbol=symbol,
        frame=frame,
        provenance=MarketDataProvenance(
            provider="fixture",
            feed="daily",
            adjustment="backward_total_return",
        ),
        retrieved_at=datetime(2024, 2, 1, tzinfo=UTC),
    )


def test_fetch_nasdaq100_snapshot_uses_official_response_shape() -> None:
    payload = {
        "data": {
            "date": "Jul 21, 2026",
            "data": {
                "rows": [
                    {"symbol": "MSFT", "companyName": "Microsoft Corporation"},
                    {"symbol": "AAPL", "companyName": "Apple Inc."},
                ]
            },
        }
    }
    requests = []

    def transport(request, timeout):
        requests.append((request, timeout))
        return _response(json.dumps(payload))

    snapshot = fetch_nasdaq100_snapshot(transport=transport)

    assert snapshot.as_of == pd.Timestamp("2026-07-21")
    assert snapshot.symbols == ["AAPL", "MSFT"]
    assert requests[0][0].full_url.endswith("/nasdaq100")
    assert requests[0][0].get_header("User-agent").startswith("Mozilla/5.0")


def test_cboe_month_download_normalizes_filters_aggregates_and_caches(tmp_path) -> None:
    csv = """Trade Date,Options Class,Underlying,Product Type,Exchange,Volume
2024/01/02,AAPL,AAPL,S,CBOE,10
2024/01/02,AAPL1,AAPL,S,BATS,3
2024/01/02,MSFT,MSFT,S,C2,7
2024/01/03,AAPL,AAPL,S,EDGX,5
"""
    requested_urls = []

    def transport(request, timeout):
        requested_urls.append(request.full_url)
        return _response(csv)

    source = CboeOptionVolumeSource(cache_dir=tmp_path, transport=transport)
    detail = source.fetch_range(["AAPL"], "2024-01-02", "2024-01-03")
    cached = source.fetch_range(["AAPL"], "2024-01-02", "2024-01-03")
    daily = aggregate_cboe_option_volume(detail)

    query = parse_qs(urlparse(requested_urls[0]).query)
    assert query["exchanges"] == list(CBOE_EXCHANGES)
    assert len(requested_urls) == 1
    pd.testing.assert_frame_equal(detail, cached)
    assert list(detail.columns) == [
        "Date",
        "OptionRoot",
        "Symbol",
        "ProductType",
        "Exchange",
        "OptionVolume",
    ]
    assert daily.loc[daily["Date"] == pd.Timestamp("2024-01-02"), "OptionVolume"].item() == 13
    assert daily.loc[0, "Coverage"] == "cboe_venues_only"


def test_price_history_eligibility_excludes_recent_listing() -> None:
    panel = MarketDataPanel.from_results(
        [
            _price_result("AAPL", pd.bdate_range("2024-01-02", periods=6)),
            _price_result("NEW", pd.bdate_range("2024-01-22", periods=3)),
        ]
    )

    eligible, report = select_price_history_eligible(
        panel,
        requested_start="2024-01-01",
        minimum_sessions=5,
        maximum_start_gap_days=10,
    )

    assert list(eligible) == ["AAPL"]
    excluded = report.set_index("symbol").loc["NEW"]
    assert not excluded["eligible"]
    assert "history_starts_after_window_tolerance" in excluded["exclusion_reason"]
    assert "insufficient_price_sessions" in excluded["exclusion_reason"]


def test_bundle_manifest_and_archive_include_market_and_option_files(tmp_path) -> None:
    universe = pd.DataFrame(
        {
            "symbol": ["AAPL", "NEW"],
            "eligible": [True, False],
            "has_cboe_option_volume": [True, False],
        }
    )
    detail = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")],
            "OptionRoot": ["AAPL"],
            "Symbol": ["AAPL"],
            "ProductType": ["S"],
            "Exchange": ["CBOE"],
            "OptionVolume": [10],
        }
    )
    daily = aggregate_cboe_option_volume(detail)
    for filename in ("ohlcv.csv", "data_quality.csv", "market_data_manifest.json"):
        (tmp_path / filename).write_text("fixture\n", encoding="ascii")
    archive = tmp_path / "bundle.tar.gz"

    result = write_free_dataset_bundle(
        tmp_path,
        universe=universe,
        option_detail=detail,
        option_daily=daily,
        start_date="2024-01-01",
        end_date="2024-03-31",
        universe_basis="current_nasdaq100_snapshot",
        archive=archive,
    )

    manifest = json.loads((tmp_path / "free_dataset_manifest.json").read_text())
    assert result == archive
    assert manifest["universe"]["symbols_in_source"] == 2
    assert manifest["universe"]["symbols_included"] == 1
    assert manifest["universe"]["eligibility"]["minimum_price_sessions"] == 252
    assert manifest["price_data"]["adjustment"] == "backward_total_return"
    assert manifest["option_data"]["symbols_with_reported_volume"] == 1
    assert "ohlcv" in manifest["files"]
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
    assert "free_nasdaq100/ohlcv.csv" in names
    assert "free_nasdaq100/cboe_option_volume_daily.csv.gz" in names
    assert "free_nasdaq100/free_dataset_manifest.json" in names
