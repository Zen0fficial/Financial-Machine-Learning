"""Normalized option observations and timing-safe daily feature construction.

The schema and feature definitions are adapted from the MIT-licensed
Financial-Machine-Learning project. See NOTICE.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, overload
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OPTION_COLUMNS = (
    "as_of",
    "trade_date",
    "underlying",
    "contract",
    "expiration",
    "strike",
    "right",
    "bid",
    "ask",
    "last_price",
    "theoretical_value",
    "volume",
    "open_interest",
    "underlying_price",
    "implied_volatility",
    "delta",
    "gamma",
    "vega",
    "provider",
)

_NEW_YORK = ZoneInfo("America/New_York")
_OPTIONAL_NUMERIC_COLUMNS = (
    "bid",
    "ask",
    "last_price",
    "theoretical_value",
    "volume",
    "open_interest",
    "underlying_price",
    "implied_volatility",
    "delta",
    "gamma",
    "vega",
)


@dataclass(frozen=True)
class OptionObservation:
    """One timestamped observation for a listed US option contract."""

    as_of: datetime
    underlying: str
    contract: str
    expiration: date
    strike: float
    right: Literal["call", "put"]
    provider: str
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = None
    theoretical_value: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    underlying_price: float | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("option as_of must be timezone-aware")
        object.__setattr__(self, "as_of", self.as_of.astimezone(UTC))

        for name in ("underlying", "contract"):
            value = str(getattr(self, name)).strip().upper()
            if not value:
                raise ValueError(f"option {name} must not be empty")
            object.__setattr__(self, name, value)
        provider = str(self.provider).strip().lower()
        if not provider:
            raise ValueError("option provider must not be empty")
        object.__setattr__(self, "provider", provider)

        right = str(self.right).strip().lower()
        if right not in {"call", "put"}:
            raise ValueError("option right must be 'call' or 'put'")
        object.__setattr__(self, "right", right)
        if self.expiration < self.trade_date:
            raise ValueError("option expiration cannot precede its observation date")

        strike = float(self.strike)
        if not isfinite(strike) or strike <= 0:
            raise ValueError("option strike must be finite and positive")
        object.__setattr__(self, "strike", strike)

        for name in (
            "bid",
            "ask",
            "last_price",
            "theoretical_value",
            "volume",
            "open_interest",
        ):
            self._normalize_optional(name, minimum=0.0)
        self._normalize_optional("underlying_price", minimum=0.0, strict=True)
        self._normalize_optional("implied_volatility", minimum=0.0, strict=True)
        self._normalize_optional("gamma", minimum=0.0)
        self._normalize_optional("vega", minimum=0.0)
        self._normalize_optional("delta", minimum=-1.05, maximum=1.05)
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError("option ask cannot be below bid")

    def _normalize_optional(
        self,
        name: str,
        *,
        minimum: float,
        maximum: float | None = None,
        strict: bool = False,
    ) -> None:
        raw = getattr(self, name)
        if raw is None:
            return
        value = float(raw)
        below = value <= minimum if strict else value < minimum
        if not isfinite(value) or below or (maximum is not None and value > maximum):
            qualifier = "positive" if strict and minimum == 0.0 else "valid"
            raise ValueError(f"option {name} must be finite and {qualifier}")
        object.__setattr__(self, name, value)

    @property
    def trade_date(self) -> date:
        return self.as_of.astimezone(_NEW_YORK).date()

    @property
    def midpoint(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0

    def to_record(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "trade_date": self.trade_date,
            "underlying": self.underlying,
            "contract": self.contract,
            "expiration": self.expiration,
            "strike": self.strike,
            "right": self.right,
            "bid": self.bid,
            "ask": self.ask,
            "last_price": self.last_price,
            "theoretical_value": self.theoretical_value,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "underlying_price": self.underlying_price,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "provider": self.provider,
        }


@dataclass(frozen=True)
class OptionChain(Sequence[OptionObservation]):
    """Strict, single-provider collection of normalized option observations."""

    observations: tuple[OptionObservation, ...]
    provider: str
    retrieved_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = str(self.provider).strip().lower()
        if not provider:
            raise ValueError("option-chain provider must not be empty")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if not self.observations:
            raise ValueError("option chain must not be empty")
        foreign = sorted(
            {item.provider for item in self.observations if item.provider != provider}
        )
        if foreign:
            raise ValueError(
                f"option chain mixes provider {provider!r} with {', '.join(foreign)}"
            )
        keys = [(item.as_of, item.contract) for item in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate option (as_of, contract) observations are not allowed")
        ordered = tuple(
            sorted(
                self.observations,
                key=lambda item: (
                    item.as_of,
                    item.underlying,
                    item.expiration,
                    item.strike,
                    item.right,
                ),
            )
        )
        object.__setattr__(self, "observations", ordered)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "retrieved_at", self.retrieved_at.astimezone(UTC))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @overload
    def __getitem__(self, index: int) -> OptionObservation: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[OptionObservation, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> OptionObservation | tuple[OptionObservation, ...]:
        return self.observations[index]

    def __iter__(self) -> Iterator[OptionObservation]:
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame.from_records(
            (item.to_record() for item in self.observations),
            columns=OPTION_COLUMNS,
        )
        frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["expiration"] = pd.to_datetime(frame["expiration"])
        return frame


def option_chain_from_frame(
    frame: pd.DataFrame,
    *,
    provider: str | None = None,
    retrieved_at: datetime | None = None,
) -> OptionChain:
    """Validate a normalized CSV/Parquet frame without dropping malformed rows."""

    required = {"as_of", "underlying", "contract", "expiration", "strike", "right"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"option frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("option frame must not be empty")
    normalized = frame.copy()
    if provider is None:
        if "provider" not in normalized:
            raise ValueError("option frame requires a provider column or explicit provider")
        providers = {
            str(value).strip().lower()
            for value in normalized["provider"].dropna().unique()
            if str(value).strip()
        }
        if len(providers) != 1:
            raise ValueError(f"option frame must contain one provider; got {sorted(providers)}")
        provider = next(iter(providers))
    provider = str(provider).strip().lower()
    if "provider" in normalized:
        embedded = {
            str(value).strip().lower()
            for value in normalized["provider"].dropna().unique()
            if str(value).strip()
        }
        if embedded and embedded != {provider}:
            raise ValueError(
                f"option frame providers {sorted(embedded)} do not match {provider!r}"
            )

    observations = []
    for row_number, row in normalized.iterrows():
        try:
            as_of = pd.Timestamp(row["as_of"])
            if as_of.tzinfo is None:
                raise ValueError("as_of must include a timezone")
            values = {
                column: _optional_float(row.get(column))
                for column in _OPTIONAL_NUMERIC_COLUMNS
            }
            observations.append(
                OptionObservation(
                    as_of=as_of.to_pydatetime(),
                    underlying=row["underlying"],
                    contract=row["contract"],
                    expiration=pd.Timestamp(row["expiration"]).date(),
                    strike=row["strike"],
                    right=row["right"],
                    provider=provider,
                    **values,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid option row {row_number}: {exc}") from exc
    return OptionChain(
        observations=tuple(observations),
        provider=provider,
        retrieved_at=retrieved_at or datetime.now(UTC),
    )


def load_option_chain(path: str | Path, *, provider: str | None = None) -> OptionChain:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    try:
        if source.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(source)
        else:
            frame = pd.read_csv(source)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support requires the optional data dependency: pip install -e '.[data]'"
        ) from exc
    retrieved_at = datetime.fromtimestamp(source.stat().st_mtime, tz=UTC)
    return option_chain_from_frame(frame, provider=provider, retrieved_at=retrieved_at)


@dataclass(frozen=True)
class OptionFeatureConfig:
    near_dte: int = 30
    far_dte: int = 90
    maximum_dte_distance: int = 21
    volume_zscore_window: int = 20

    def __post_init__(self) -> None:
        for name in (
            "near_dte",
            "far_dte",
            "maximum_dte_distance",
            "volume_zscore_window",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.far_dte <= self.near_dte:
            raise ValueError("far_dte must be greater than near_dte")


def build_daily_option_features(
    data: OptionChain | pd.DataFrame,
    config: OptionFeatureConfig | None = None,
) -> pd.DataFrame:
    """Aggregate normalized contracts to one row per trade date and underlying."""

    cfg = config or OptionFeatureConfig()
    frame = (
        data.to_frame()
        if isinstance(data, OptionChain)
        else option_chain_from_frame(data).to_frame()
    )
    required = {
        "trade_date",
        "underlying",
        "expiration",
        "strike",
        "right",
        "volume",
        "open_interest",
        "underlying_price",
        "implied_volatility",
        "delta",
        "bid",
        "ask",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"option frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("option frame must not be empty")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["expiration"] = pd.to_datetime(frame["expiration"]).dt.normalize()
    frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True)
    frame["underlying"] = frame["underlying"].astype(str).str.upper()
    frame["right"] = frame["right"].astype(str).str.lower()
    for column in required.difference(
        {"trade_date", "expiration", "underlying", "right"}
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.sort_values("as_of")
        .drop_duplicates(
            subset=["trade_date", "underlying", "contract"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    records = [
        _aggregate_chain(trade_date, underlying, chain, cfg)
        for (trade_date, underlying), chain in frame.groupby(
            ["trade_date", "underlying"], sort=True
        )
    ]
    result = pd.DataFrame.from_records(records).set_index(["trade_date", "underlying"])
    return add_option_feature_dynamics(result, window=cfg.volume_zscore_window)


def _aggregate_chain(
    trade_date: pd.Timestamp,
    underlying: str,
    chain: pd.DataFrame,
    config: OptionFeatureConfig,
) -> dict[str, Any]:
    calls = chain[chain["right"].eq("call")]
    puts = chain[chain["right"].eq("put")]
    call_volume = float(calls["volume"].fillna(0.0).sum())
    put_volume = float(puts["volume"].fillna(0.0).sum())
    call_oi = float(calls["open_interest"].fillna(0.0).sum())
    put_oi = float(puts["open_interest"].fillna(0.0).sum())

    quoted = chain.loc[
        chain["bid"].gt(0.0) & chain["ask"].gt(0.0) & chain["ask"].ge(chain["bid"])
    ].copy()
    midpoint = quoted["bid"].add(quoted["ask"]).div(2.0)
    relative_spread = quoted["ask"].sub(quoted["bid"]).div(midpoint)

    near = _nearest_expiry(chain, trade_date, config.near_dte, config.maximum_dte_distance)
    far = _nearest_expiry(chain, trade_date, config.far_dte, config.maximum_dte_distance)
    near_atm = _atm_iv(near)
    far_atm = _atm_iv(far)
    return {
        "trade_date": trade_date,
        "underlying": underlying,
        "latest_as_of": chain["as_of"].max(),
        "contract_count": int(len(chain)),
        "quoted_contract_count": int(len(quoted)),
        "call_volume": call_volume,
        "put_volume": put_volume,
        "option_volume": call_volume + put_volume,
        "put_call_volume_ratio": _safe_ratio(put_volume, call_volume),
        "call_open_interest": call_oi,
        "put_open_interest": put_oi,
        "open_interest": call_oi + put_oi,
        "put_call_open_interest_ratio": _safe_ratio(put_oi, call_oi),
        "median_relative_spread": (
            float(relative_spread.median()) if not relative_spread.empty else np.nan
        ),
        f"atm_iv_{config.near_dte}d": near_atm,
        f"atm_iv_{config.far_dte}d": far_atm,
        f"iv_term_slope_{config.far_dte}_{config.near_dte}": (
            far_atm - near_atm
            if np.isfinite(far_atm) and np.isfinite(near_atm)
            else np.nan
        ),
        f"put_call_skew_{config.near_dte}d": _delta_skew(near),
    }


def _nearest_expiry(
    chain: pd.DataFrame,
    trade_date: pd.Timestamp,
    target_dte: int,
    maximum_distance: int,
) -> pd.DataFrame:
    usable = chain.loc[
        chain["implied_volatility"].gt(0.0)
        & chain["underlying_price"].gt(0.0)
        & chain["expiration"].ge(trade_date)
    ].copy()
    if usable.empty:
        return usable
    usable["dte"] = (usable["expiration"] - trade_date).dt.days
    expiry_dtes = usable.groupby("expiration")["dte"].first()
    selected_expiry = (expiry_dtes - target_dte).abs().idxmin()
    if abs(int(expiry_dtes.loc[selected_expiry]) - target_dte) > maximum_distance:
        return usable.iloc[0:0]
    return usable[usable["expiration"].eq(selected_expiry)]


def _atm_iv(chain: pd.DataFrame) -> float:
    if chain.empty:
        return np.nan
    values = []
    for _, side in chain.groupby("right"):
        moneyness = np.log(side["strike"].div(side["underlying_price"]))
        position = moneyness.abs().idxmin()
        value = float(side.loc[position, "implied_volatility"])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def _delta_skew(chain: pd.DataFrame) -> float:
    usable = chain.loc[chain["delta"].notna() & chain["implied_volatility"].gt(0.0)]
    calls = usable[usable["right"].eq("call")]
    puts = usable[usable["right"].eq("put")]
    if calls.empty or puts.empty:
        return np.nan
    call_index = calls["delta"].sub(0.25).abs().idxmin()
    put_index = puts["delta"].add(0.25).abs().idxmin()
    return float(
        puts.loc[put_index, "implied_volatility"]
        - calls.loc[call_index, "implied_volatility"]
    )


def add_option_feature_dynamics(features: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Add changes and volume surprise using only earlier days as the baseline."""

    if window < 2:
        raise ValueError("window must be at least 2")
    result = features.sort_index().copy()
    if not isinstance(result.index, pd.MultiIndex) or result.index.nlevels != 2:
        raise ValueError("features must be indexed by (trade_date, underlying)")
    grouped = result.groupby(level="underlying", group_keys=False)
    change_columns = [
        "put_call_volume_ratio",
        "put_call_open_interest_ratio",
        *[column for column in result if column.startswith("atm_iv_")],
        *[column for column in result if column.startswith("put_call_skew_")],
    ]
    for column in dict.fromkeys(change_columns):
        if column in result:
            result[f"{column}_change_1d"] = grouped[column].diff()

    log_volume = np.log1p(result["option_volume"])
    history = log_volume.groupby(level="underlying").shift(1)
    history_grouped = history.groupby(level="underlying")
    mean = history_grouped.transform(
        lambda values: values.rolling(window, min_periods=window).mean()
    )
    standard_deviation = history_grouped.transform(
        lambda values: values.rolling(window, min_periods=window).std()
    )
    result[f"option_volume_zscore_{window}d"] = log_volume.sub(mean).div(
        standard_deviation.replace(0.0, np.nan)
    )
    return result


def make_option_supervised_frame(
    option_features: pd.DataFrame,
    close_prices: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 5),
    execution_lag: int = 1,
    realized_volatility_window: int = 20,
    drop_missing_targets: bool = True,
) -> pd.DataFrame:
    """Align end-of-day option features to returns beginning after an execution lag."""

    if execution_lag < 1:
        raise ValueError("execution_lag must be at least 1 for EOD option features")
    if realized_volatility_window < 2:
        raise ValueError("realized_volatility_window must be at least 2")
    horizon_values = tuple(int(value) for value in horizons)
    if not horizon_values or any(value < 1 for value in horizon_values):
        raise ValueError("horizons must contain positive integers")
    if not isinstance(option_features.index, pd.MultiIndex):
        raise ValueError("option_features must use a (trade_date, underlying) index")
    prices = close_prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index().sort_index(axis="columns").astype(float)
    if prices.index.has_duplicates:
        raise ValueError("close_prices index must be unique")

    returns = np.log(prices).diff()
    realized_volatility = (
        returns.rolling(
            realized_volatility_window,
            min_periods=realized_volatility_window,
        ).std()
        * np.sqrt(252.0)
    )
    per_asset = []
    for underlying in option_features.index.get_level_values("underlying").unique():
        if underlying not in prices.columns:
            continue
        asset_features = option_features.xs(underlying, level="underlying").copy()
        asset_features[f"realized_volatility_{realized_volatility_window}d"] = (
            realized_volatility[underlying].reindex(asset_features.index)
        )
        near_iv = next(
            (column for column in asset_features if column.startswith("atm_iv_")),
            None,
        )
        if near_iv:
            asset_features[f"iv_realized_spread_{near_iv.removeprefix('atm_iv_')}"] = (
                asset_features[near_iv].sub(
                    asset_features[f"realized_volatility_{realized_volatility_window}d"]
                )
            )
        for horizon in horizon_values:
            entry = prices[underlying].shift(-execution_lag)
            exit_price = prices[underlying].shift(-(execution_lag + horizon))
            asset_features[f"target_return_{horizon}d"] = exit_price.div(entry).sub(
                1.0
            ).reindex(asset_features.index)
        asset_features["underlying"] = underlying
        per_asset.append(asset_features.reset_index())
    if not per_asset:
        raise ValueError("no option underlyings matched close-price columns")
    result = pd.concat(per_asset, ignore_index=True).set_index(
        ["trade_date", "underlying"]
    ).sort_index()
    if drop_missing_targets:
        target_columns = [f"target_return_{value}d" for value in horizon_values]
        result = result.dropna(subset=target_columns, how="any")
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value) or value == "":
        return None
    return float(value)
