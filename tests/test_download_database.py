import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts import download_database


class FakeMinuteSource:
    def __init__(self) -> None:
        self.calls = []

    def fetch_result(self, symbol: str, start: str, end: str, refresh: bool = False):
        self.calls.append((symbol, start, end, refresh))
        return SimpleNamespace(frame=[1, 2, 3])


def test_minute_phase_uses_dispatcher_argument_order() -> None:
    source = FakeMinuteSource()
    logger = logging.getLogger("test_minute_phase_uses_dispatcher_argument_order")
    logger.handlers[:] = [logging.NullHandler()]
    logger.propagate = False

    phase = download_database.run_phase(
        "minute",
        download_database.fetch_symbol_minute,
        ["AAPL"],
        1,
        logger,
        source,
        "2024-01-02",
        "2024-01-05",
    )

    assert phase["ok"] == 1
    assert phase["failed"] == 0
    assert source.calls == [("AAPL", "2024-01-02", "2024-01-05", False)]


def test_load_symbols_excludes_explicit_symbols() -> None:
    args = SimpleNamespace(
        symbols="AAPL,ANSS,MSFT",
        universe_file=None,
        universe="custom",
        exclude_symbols="ANSS",
        listed_since="2024-01-01",
        end="2026-07-24",
        respect_source_exclusions=False,
    )

    assert download_database.load_symbols(args) == ["AAPL", "MSFT"]


def test_universe_table_tracks_listing_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "universe.csv"
        pd.DataFrame(
            [
                {"symbol": "AAA", "listed_date": "2023-01-01", "delisted_date": ""},
                {"symbol": "BBB", "listed_date": "2024-05-01", "delisted_date": ""},
                {"symbol": "CCC", "listed_date": "2020-01-01", "delisted_date": "2025-07-17"},
                {"symbol": "OLD", "listed_date": "2010-01-01", "delisted_date": "2023-12-31"},
            ]
        ).to_csv(path, index=False)
        args = SimpleNamespace(
            symbols="all",
            universe_file=str(path),
            universe="sp500",
            exclude_symbols="",
            listed_since="2024-01-01",
            end="2026-07-24",
            respect_source_exclusions=False,
        )

        table = download_database.build_universe_table(args)

    included = table.loc[table["included"], "symbol"].tolist()
    assert included == ["AAA", "BBB", "CCC"]
    assert table.loc[table["symbol"] == "BBB", "listed_after_period_start"].item() is True
    assert table.loc[table["symbol"] == "CCC", "delisted_during_period"].item() is True
    assert table.loc[table["symbol"] == "OLD", "exclusion_reason"].item() == "not_listed_during_period"


def test_symbol_windows_clamp_to_listing_lifecycle() -> None:
    universe = pd.DataFrame(
        [
            {"symbol": "AAA", "listed_date": "", "delisted_date": "", "included": True},
            {
                "symbol": "NEW",
                "listed_date": "2024-05-01",
                "delisted_date": "",
                "included": True,
            },
            {
                "symbol": "OLD",
                "listed_date": "",
                "delisted_date": "2025-07-17",
                "included": True,
            },
        ]
    )

    windows = download_database.build_symbol_windows(
        universe, "2024-01-02", "2026-07-24"
    )

    assert windows == {
        "AAA": ("2024-01-02", "2026-07-24"),
        "NEW": ("2024-05-01", "2026-07-24"),
        "OLD": ("2024-01-02", "2025-07-17"),
    }


def test_daily_coverage_windows_adjust_only_actual_start() -> None:
    logger = logging.getLogger("test_daily_coverage_windows")
    logger.handlers[:] = [logging.NullHandler()]
    logger.propagate = False

    with tempfile.TemporaryDirectory() as tmpdir:
        daily_dir = Path(tmpdir)
        first_daily_date = "2024-03-12"
        pd.DataFrame(
            [
                {"Date": first_daily_date, "Close": 39.62},
                {"Date": "2024-03-13", "Close": 40.10},
            ]
        ).to_csv(
            daily_dir / "polygon_1d_backward_total_return_XYZ_20240102_20260724.csv",
            index=False,
        )

        adjusted = download_database.build_daily_coverage_windows(
            ["XYZ", "AAPL"],
            {
                "XYZ": ("2024-01-02", "2026-07-24"),
                "AAPL": ("2024-01-02", "2026-07-24"),
            },
            daily_dir,
            logger,
        )

    assert adjusted["XYZ"] == (first_daily_date, "2026-07-24")
    assert adjusted["AAPL"] == ("2024-01-02", "2026-07-24")


def test_ticker_details_aligns_columns_and_uses_historical_fallback() -> None:
    logger = logging.getLogger("test_ticker_details_aligns_columns")
    logger.handlers[:] = [logging.NullHandler()]
    logger.propagate = False

    with tempfile.TemporaryDirectory() as tmpdir:
        source = SimpleNamespace(cache_dir=Path(tmpdir), base_url="https://example.test")
        responses = [
            {"results": {"ticker": "AAPL", "name": "Apple", "homepage_url": "https://apple.com"}},
            download_database.MarketDataError("not found"),
            {"results": {"ticker": "OLD", "name": "Old Company", "sic_code": "1234"}},
        ]
        with patch.object(download_database, "_fetch_single", side_effect=responses) as fetch:
            first = download_database.fetch_ticker_details(source, "AAPL", logger)
            second = download_database.fetch_ticker_details(
                source, "OLD", logger, reference_date="2024-03-18"
            )

        frame = pd.read_csv(Path(tmpdir) / "ticker_details.csv")

    assert first[0] is True
    assert second[0] is True
    assert frame["ticker"].tolist() == ["AAPL", "OLD"]
    assert set(["homepage_url", "sic_code", "reference_date"]).issubset(frame.columns)
    assert frame.loc[frame["ticker"] == "OLD", "reference_date"].item() == "2024-03-15"
    assert fetch.call_args_list[-1].args[1].endswith("?date=2024-03-15")
