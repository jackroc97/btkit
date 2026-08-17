"""
Unit tests confirming LegConfig.dte_skip_weekends is threaded into the
leg_specs dicts EntryScanner._select_legs() passes to greeks_for_all_legs(),
for both the simple (fast-path) and stepped selection modes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from unittest.mock import MagicMock

import polars as pl
import pytest
from pydantic import ValidationError

from btkit.backtest.entry import EntryScanner
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    SteppedLegConfig,
    SteppedStep,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)


class _CapturingDB:
    """Records the leg_specs passed to greeks_for_all_legs; returns empty."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def greeks_for_all_legs(self, ts_event_underlying, leg_specs, **kwargs) -> pl.DataFrame:
        self.calls.append(leg_specs)
        return pl.DataFrame()

    def greeks_for_strike_legs(self, *a, **k) -> pl.DataFrame:
        return pl.DataFrame()


def _candidates() -> pl.DataFrame:
    ts = datetime(2024, 1, 4, 15, 0, tzinfo=UTC)  # a Thursday
    return pl.DataFrame(
        {
            "ts_event": pl.Series([ts], dtype=pl.Datetime("us", "UTC")),
            "underlying_id": pl.Series([100], dtype=pl.Int64),
            "open": pl.Series([5000.0], dtype=pl.Float64),
            "high": pl.Series([5010.0], dtype=pl.Float64),
            "low": pl.Series([4990.0], dtype=pl.Float64),
            "close": pl.Series([5005.0], dtype=pl.Float64),
            "volume": pl.Series([1], dtype=pl.Int64),
            "iv": pl.Series([0.5], dtype=pl.Float64),
        }
    )


def _scanner(leg: LegConfig) -> EntryScanner:
    trade = TradeDefinition(
        name="t",
        instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
        entry=EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0))),
        legs=[leg],
        exit=ExitConfig(stop_loss=2.0),
    )
    strategy = StrategyDefinition(
        name="test",
        universe=UniverseConfig(start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)),
        trades=[trade],
    )
    return EntryScanner(MagicMock(), strategy, trade)


class TestDteSkipWeekendsPlumbing:
    def test_simple_path_propagates_true(self):
        leg = LegConfig(
            name="sp",
            right="put",
            action="sell_to_open",
            dte={"target": 2, "tolerance": 0},
            delta={"target": -0.10},
            dte_skip_weekends=True,
        )
        db = _CapturingDB()
        scanner = _scanner(leg)
        scanner.db = db

        scanner._select_legs(_candidates())

        assert db.calls, "expected greeks_for_all_legs to be called"
        specs = db.calls[0]
        assert specs[0]["dte_skip_weekends"] is True

    def test_simple_path_default_false(self):
        leg = LegConfig(
            name="sp",
            right="put",
            action="sell_to_open",
            dte={"target": 2, "tolerance": 0},
            delta={"target": -0.10},
        )
        db = _CapturingDB()
        scanner = _scanner(leg)
        scanner.db = db

        scanner._select_legs(_candidates())

        specs = db.calls[0]
        assert specs[0]["dte_skip_weekends"] is False

    def test_stepped_path_propagates_true(self):
        leg = LegConfig(
            name="sp",
            right="put",
            action="sell_to_open",
            dte_skip_weekends=True,
            stepped=SteppedLegConfig(
                source="iv", steps=[SteppedStep(dte=2, delta=-0.10)]
            ),
        )
        db = _CapturingDB()
        scanner = _scanner(leg)
        scanner.db = db

        scanner._select_legs(_candidates())

        assert db.calls, "expected greeks_for_all_legs to be called"
        specs = db.calls[0]
        assert specs[0]["dte_skip_weekends"] is True

    def test_dte_skip_weekends_rejects_strike_offset(self):
        with pytest.raises(ValidationError, match="dte_skip_weekends has no effect"):
            LegConfig(
                name="lp",
                right="put",
                action="buy_to_open",
                strike_offset=-50.0,
                reference_leg="sp",
                dte_skip_weekends=True,
            )
