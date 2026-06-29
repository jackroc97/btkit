"""
Unit tests for the roll block feature.

Covers:
  - RollConfig validation (at_least_one_trigger)
  - roll field default on TradeDefinition
  - _enforce_max_entries_per_day roll re-entry bypass
  - roll + no_reentry_after_loss interaction (roll does NOT block re-entry)
"""
from __future__ import annotations

from datetime import UTC, date, datetime, time
from unittest.mock import MagicMock

import polars as pl
import pytest

from btkit.backtest.engine import BacktestEngine
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    IntSweep,
    LegConfig,
    RollConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TZ = "America/New_York"


def _make_engine(max_entries_per_day=None, no_reentry=False, roll=None):
    entry_cfg = EntryConfig(
        window=EntryWindowConfig(start=time(9, 30), end=time(16, 0)),
        max_entries_per_day=max_entries_per_day,
        no_reentry_after_loss=no_reentry,
    )
    strategy = StrategyDefinition(
        name="test",
        universe=UniverseConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 12, 31),
        ),
        trades=[
            TradeDefinition(
                name="t",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=entry_cfg,
                legs=[
                    LegConfig(name="short", right="put", action="sell_to_open", dte=0, delta=-0.16),
                    LegConfig(name="long", right="put", action="buy_to_open", dte=0, strike_offset=-50.0, reference_leg="short"),
                ],
                exit=ExitConfig(stop_loss=1.5, take_profit=0.5),
                roll=roll,
            )
        ],
    )
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.strategy = strategy
    return engine


def _ts(dt_str: str) -> datetime:
    import zoneinfo
    naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    return naive.replace(tzinfo=zoneinfo.ZoneInfo(TZ)).astimezone(UTC)


def _make_frames(rows: list[dict]) -> tuple[pl.DataFrame, pl.DataFrame]:
    entries = pl.DataFrame({
        "entry_id":   [r["entry_id"] for r in rows],
        "entry_time": pl.Series([_ts(r["entry_time"]) for r in rows]).dt.cast_time_unit("us"),
    })
    exits = pl.DataFrame({
        "entry_id":    [r["entry_id"] for r in rows],
        "exit_reason": [r["exit_reason"] for r in rows],
        "exit_time":   pl.Series([_ts(r["exit_time"]) for r in rows]).dt.cast_time_unit("us"),
    })
    return entries, exits


# ---------------------------------------------------------------------------
# RollConfig validation tests
# ---------------------------------------------------------------------------

class TestRollConfig:

    def test_dte_only_valid(self):
        cfg = RollConfig(dte=10)
        assert cfg.dte == 10
        assert cfg.vega is None

    def test_vega_only_valid(self):
        cfg = RollConfig(vega=0.15)
        assert cfg.vega == 0.15
        assert cfg.dte is None

    def test_both_dte_and_vega_valid(self):
        cfg = RollConfig(dte=10, vega=0.15)
        assert cfg.dte == 10
        assert cfg.vega == 0.15

    def test_neither_dte_nor_vega_raises(self):
        with pytest.raises(Exception, match="roll requires at least one trigger"):
            RollConfig()

    def test_custom_window(self):
        cfg = RollConfig(dte=7, window=EntryWindowConfig(start=time(9, 45), end=time(14, 30)))
        assert cfg.window is not None
        assert cfg.window.start == time(9, 45)

    def test_window_defaults_none(self):
        cfg = RollConfig(dte=5)
        assert cfg.window is None


class TestTradeDefinitionRoll:

    def test_roll_defaults_none(self):
        trade = TradeDefinition(
            name="t",
            instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
            entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0))),
            legs=[
                LegConfig(name="short", right="put", action="sell_to_open", dte=0, delta=-0.16),
                LegConfig(name="long", right="put", action="buy_to_open", dte=0, strike_offset=-50.0, reference_leg="short"),
            ],
            exit=ExitConfig(),
        )
        assert trade.roll is None

    def test_roll_accepts_roll_config(self):
        trade = TradeDefinition(
            name="t",
            instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
            entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0))),
            legs=[
                LegConfig(name="short", right="put", action="sell_to_open", dte=0, delta=-0.16),
                LegConfig(name="long", right="put", action="buy_to_open", dte=0, strike_offset=-50.0, reference_leg="short"),
            ],
            exit=ExitConfig(),
            roll=RollConfig(dte=10),
        )
        assert trade.roll is not None
        assert trade.roll.dte == 10


# ---------------------------------------------------------------------------
# _enforce_max_entries_per_day roll bypass tests
# ---------------------------------------------------------------------------

class TestRollReentryBypass:

    def test_roll_reentry_not_counted_against_daily_cap(self):
        """
        Scenario: max_entries_per_day=1 with a roll.
        Day: entry at 09:45 → rolls at 11:00 → re-enters at 11:01
        Without bypass: only the 09:45 entry survives.
        With bypass: both 09:45 and 11:01 survive (11:01 is the roll re-entry).
        """
        engine = _make_engine(max_entries_per_day=1)
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:45", "exit_time": "2024-01-02 11:00", "exit_reason": "roll"},
            {"entry_id": 2, "entry_time": "2024-01-02 11:01", "exit_time": "2024-01-02 14:00", "exit_reason": "dte_exit"},
        ]
        entries, exits = _make_frames(rows)
        roll_exit_ids = {1}
        out_entries, out_exits = engine._enforce_max_entries_per_day(
            entries, exits, max_entries=1, roll_exit_ids=roll_exit_ids
        )
        assert sorted(out_entries["entry_id"].to_list()) == [1, 2]

    def test_non_roll_second_entry_still_capped(self):
        """A regular re-entry (not following a roll) is still subject to the daily cap."""
        engine = _make_engine(max_entries_per_day=1)
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:45", "exit_time": "2024-01-02 11:00", "exit_reason": "take_profit"},
            {"entry_id": 2, "entry_time": "2024-01-02 11:01", "exit_time": "2024-01-02 14:00", "exit_reason": "dte_exit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_max_entries_per_day(entries, exits, max_entries=1)
        assert sorted(out_entries["entry_id"].to_list()) == [1]

    def test_roll_reentry_on_different_day_is_not_exempt(self):
        """A roll exit on day D does not exempt an entry on day D+1 from the cap."""
        engine = _make_engine(max_entries_per_day=1)
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:45", "exit_time": "2024-01-02 15:30", "exit_reason": "roll"},
            {"entry_id": 2, "entry_time": "2024-01-03 09:45", "exit_time": "2024-01-03 14:00", "exit_reason": "dte_exit"},
            {"entry_id": 3, "entry_time": "2024-01-03 11:00", "exit_time": "2024-01-03 15:00", "exit_reason": "expiry"},
        ]
        entries, exits = _make_frames(rows)
        # Roll happened on Jan 2; re-entry on Jan 3 09:45 is fresh, not a roll re-entry
        out_entries, _ = engine._enforce_max_entries_per_day(
            entries, exits, max_entries=1, roll_exit_ids={1}
        )
        # Jan 3: both 2 and 3 are on the same day; only first (rank ≤ 1) survives → id=2
        assert sorted(out_entries["entry_id"].to_list()) == [1, 2]

    def test_no_roll_exit_ids_behaves_as_before(self):
        """Passing roll_exit_ids=None falls back to standard daily cap behavior."""
        engine = _make_engine(max_entries_per_day=1)
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:45", "exit_time": "2024-01-02 11:00", "exit_reason": "take_profit"},
            {"entry_id": 2, "entry_time": "2024-01-02 11:30", "exit_time": "2024-01-02 14:00", "exit_reason": "expiry"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_max_entries_per_day(
            entries, exits, max_entries=1, roll_exit_ids=None
        )
        assert sorted(out_entries["entry_id"].to_list()) == [1]

    def test_empty_entries_with_roll_exit_ids(self):
        """Empty inputs do not raise even when roll_exit_ids is provided."""
        engine = _make_engine(max_entries_per_day=1)
        entries = pl.DataFrame({
            "entry_id": pl.Series([], dtype=pl.Int64),
            "entry_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
        })
        exits = pl.DataFrame({
            "entry_id": pl.Series([], dtype=pl.Int64),
            "exit_reason": pl.Series([], dtype=pl.String),
            "exit_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
        })
        out_entries, out_exits = engine._enforce_max_entries_per_day(
            entries, exits, max_entries=1, roll_exit_ids={99}
        )
        assert out_entries.is_empty()
        assert out_exits.is_empty()


class TestRollNoReentryInteraction:

    def test_roll_exit_does_not_block_reentry(self):
        """
        no_reentry_after_loss only blocks re-entries after stop_loss/gap_sl.
        A roll exit reason does not trigger the block.
        """
        engine = _make_engine(no_reentry=True)
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:45", "exit_reason": "roll"},
            {"entry_id": 2, "entry_time": "2024-01-02 11:01", "exit_reason": "expiry"},
        ]
        entries = pl.DataFrame({
            "entry_id":   [r["entry_id"] for r in rows],
            "entry_time": pl.Series([_ts(r["entry_time"]) for r in rows]).dt.cast_time_unit("us"),
        })
        exits = pl.DataFrame({
            "entry_id":    [r["entry_id"] for r in rows],
            "exit_reason": [r["exit_reason"] for r in rows],
        })
        out_entries, _ = engine._enforce_no_reentry_after_loss(entries, exits)
        assert sorted(out_entries["entry_id"].to_list()) == [1, 2]
