from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from us_factor_screening.options import (
    OPTION_COLUMNS,
    OptionChain,
    OptionObservation,
    build_daily_option_features,
    make_option_supervised_frame,
    option_chain_from_frame,
)


def _observation(**overrides) -> OptionObservation:
    values = {
        "as_of": datetime(2024, 1, 3, 20, 46, tzinfo=UTC),
        "underlying": "aapl",
        "contract": "AAPL240202C00100000",
        "expiration": date(2024, 2, 2),
        "strike": 100,
        "right": "call",
        "provider": "fixture",
        "bid": 4.9,
        "ask": 5.1,
        "volume": 10,
        "open_interest": 100,
        "underlying_price": 100,
        "implied_volatility": 0.25,
        "delta": 0.5,
    }
    values.update(overrides)
    return OptionObservation(**values)


def _chain(days: int = 25) -> OptionChain:
    observations = []
    start = date(2024, 1, 2)
    for offset in range(days):
        trade_date = start + timedelta(days=offset)
        as_of = datetime.combine(
            trade_date,
            time(15, 46),
            tzinfo=ZoneInfo("America/New_York"),
        )
        volume_scale = offset + 1
        contracts = (
            ("C100", "call", 100, 0.50, 0.20, 10 * volume_scale, 100),
            ("P100", "put", 100, -0.50, 0.22, 20 * volume_scale, 200),
            ("C110", "call", 110, 0.25, 0.21, 5 * volume_scale, 50),
            ("P090", "put", 90, -0.25, 0.30, 8 * volume_scale, 80),
        )
        for suffix, right, strike, delta, iv, volume, open_interest in contracts:
            observations.append(
                OptionObservation(
                    as_of=as_of,
                    underlying="TEST",
                    contract=f"TEST{trade_date:%Y%m%d}{suffix}",
                    expiration=trade_date + timedelta(days=30),
                    strike=strike,
                    right=right,
                    bid=1.0,
                    ask=1.1,
                    volume=volume,
                    open_interest=open_interest,
                    underlying_price=100,
                    implied_volatility=iv,
                    delta=delta,
                    provider="fixture",
                )
            )
        for right, delta, iv in (("call", 0.5, 0.24), ("put", -0.5, 0.26)):
            observations.append(
                OptionObservation(
                    as_of=as_of,
                    underlying="TEST",
                    contract=f"TEST{trade_date:%Y%m%d}{right[0].upper()}FAR",
                    expiration=trade_date + timedelta(days=90),
                    strike=100,
                    right=right,
                    bid=2.0,
                    ask=2.1,
                    volume=1,
                    open_interest=10,
                    underlying_price=100,
                    implied_volatility=iv,
                    delta=delta,
                    provider="fixture",
                )
            )
    return OptionChain(
        observations=tuple(observations),
        provider="fixture",
        retrieved_at=observations[-1].as_of,
    )


def test_option_chain_exports_schema_and_rejects_duplicates() -> None:
    observation = _observation()
    chain = OptionChain(
        observations=(observation,),
        provider="fixture",
        retrieved_at=datetime(2024, 1, 3, 21, tzinfo=UTC),
    )

    assert tuple(chain.to_frame().columns) == OPTION_COLUMNS
    assert chain[0].midpoint == pytest.approx(5.0)
    with pytest.raises(ValueError, match="duplicate option"):
        OptionChain(
            observations=(observation, observation),
            provider="fixture",
            retrieved_at=datetime(2024, 1, 3, 21, tzinfo=UTC),
        )


def test_option_frame_requires_timestamp_and_single_provider() -> None:
    frame = OptionChain(
        observations=(_observation(),),
        provider="fixture",
        retrieved_at=datetime(2024, 1, 3, 21, tzinfo=UTC),
    ).to_frame()
    loaded = option_chain_from_frame(frame)
    assert loaded.provider == "fixture"

    frame["as_of"] = pd.Series(["2024-01-03 15:46:00"], dtype="object")
    with pytest.raises(ValueError, match="timezone"):
        option_chain_from_frame(frame)


def test_daily_features_use_prior_history_for_volume_surprise() -> None:
    features = build_daily_option_features(_chain())
    first = features.iloc[0]

    assert first["put_call_volume_ratio"] == pytest.approx(29 / 16)
    assert first["atm_iv_30d"] == pytest.approx(0.21)
    assert first["atm_iv_90d"] == pytest.approx(0.25)
    assert first["put_call_skew_30d"] == pytest.approx(0.09)
    assert np.isnan(features["option_volume_zscore_20d"].iloc[19])
    assert np.isfinite(features["option_volume_zscore_20d"].iloc[20])


def test_daily_features_use_latest_contract_snapshot() -> None:
    chain = _chain(days=1)
    first_contract = chain[0]
    earlier = OptionObservation(
        **{
            **first_contract.__dict__,
            "as_of": first_contract.as_of - timedelta(minutes=30),
            "volume": 1_000,
        }
    )
    with_earlier_snapshot = OptionChain(
        observations=(earlier, *chain.observations),
        provider="fixture",
        retrieved_at=chain.retrieved_at,
    )

    features = build_daily_option_features(with_earlier_snapshot)

    assert features.iloc[0]["call_volume"] == 16


def test_option_targets_begin_after_execution_lag() -> None:
    features = build_daily_option_features(_chain())
    dates = pd.date_range("2024-01-02", periods=35, freq="D")
    prices = pd.DataFrame({"TEST": np.arange(100.0, 135.0)}, index=dates)

    supervised = make_option_supervised_frame(
        features,
        prices,
        horizons=(1, 5),
        execution_lag=1,
        realized_volatility_window=5,
        drop_missing_targets=False,
    )
    first = supervised.iloc[0]

    assert first["target_return_1d"] == pytest.approx(102.0 / 101.0 - 1.0)
    assert first["target_return_5d"] == pytest.approx(106.0 / 101.0 - 1.0)
