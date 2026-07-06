"""
Unit tests for BacktestEngine._enforce_no_reentry_after_loss.

Tests call the method directly with hand-constructed DataFrames.
No DB access, no actual backtest run.

Terminology:
  - "loss"  = exit_reason in {"stop_loss", "gap_sl"}
  - "win"   = any other exit_reason (take_profit, expiry, dte, …)
  - Day D is "blocked" once a loss on day D has been seen; subsequent
    entries on day D are dropped.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import polars as pl

from btkit.backtest.engine import BacktestEngine
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TZ = "America/New_York"


def _make_engine() -> BacktestEngine:
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
                entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0))),
                legs=[
                    LegConfig(
                        name="short",
                        right="put",
                        action="sell_to_open",
                        dte=0,
                        delta={"target": -0.16},
                    ),
                    LegConfig(
                        name="long",
                        right="put",
                        action="buy_to_open",
                        dte=0,
                        strike_offset=-50.0,
                        reference_leg="short",
                    ),
                ],
                exit=ExitConfig(stop_loss=1.5, take_profit=0.5),
            )
        ],
    )
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.strategy = strategy
    return engine


def _ts(dt_str: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM' as Eastern and return UTC datetime."""
    import zoneinfo

    naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    return naive.replace(tzinfo=zoneinfo.ZoneInfo(TZ)).astimezone(UTC)


def _make_frames(rows: list[dict]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build entries and exits DataFrames from a list of dicts with keys:
        entry_id, entry_time (str), exit_reason (str)
    """
    entries = pl.DataFrame(
        {
            "entry_id": [r["entry_id"] for r in rows],
            "entry_time": pl.Series([_ts(r["entry_time"]) for r in rows]).dt.cast_time_unit("us"),
        }
    )
    exits = pl.DataFrame(
        {
            "entry_id": [r["entry_id"] for r in rows],
            "exit_reason": [r["exit_reason"] for r in rows],
        }
    )
    return entries, exits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoReentryAfterLoss:
    def test_no_loss_no_filtering(self):
        """When no positions are losses, all entries survive."""
        engine = _make_engine()
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:30", "exit_reason": "take_profit"},
            {"entry_id": 2, "entry_time": "2024-01-02 11:00", "exit_reason": "take_profit"},
            {"entry_id": 3, "entry_time": "2024-01-03 09:30", "exit_reason": "expiry"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, out_exits = engine._enforce_no_reentry_after_loss(entries, exits)
        assert sorted(out_entries["entry_id"].to_list()) == [1, 2, 3]
        assert sorted(out_exits["entry_id"].to_list()) == [1, 2, 3]

    def test_loss_blocks_subsequent_same_day_entry(self):
        """Re-entry after a stop-loss on the same calendar day is dropped."""
        engine = _make_engine()
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:30", "exit_reason": "stop_loss"},
            {"entry_id": 2, "entry_time": "2024-01-02 11:00", "exit_reason": "take_profit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_no_reentry_after_loss(entries, exits)
        assert out_entries["entry_id"].to_list() == [1]

    def test_gap_sl_also_blocks_reentry(self):
        """gap_sl is treated as a loss and blocks same-day re-entry."""
        engine = _make_engine()
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:30", "exit_reason": "gap_sl"},
            {"entry_id": 2, "entry_time": "2024-01-02 11:00", "exit_reason": "take_profit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_no_reentry_after_loss(entries, exits)
        assert out_entries["entry_id"].to_list() == [1]

    def test_loss_does_not_block_next_day(self):
        """A stop-loss on day D does not block entries on day D+1."""
        engine = _make_engine()
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:30", "exit_reason": "stop_loss"},
            {"entry_id": 2, "entry_time": "2024-01-03 09:30", "exit_reason": "take_profit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_no_reentry_after_loss(entries, exits)
        assert sorted(out_entries["entry_id"].to_list()) == [1, 2]

    def test_loss_kept_reentry_dropped(self):
        """The losing position itself is kept; only the subsequent re-entry is dropped."""
        engine = _make_engine()
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:30", "exit_reason": "stop_loss"},
            {"entry_id": 2, "entry_time": "2024-01-02 12:00", "exit_reason": "take_profit"},
            {"entry_id": 3, "entry_time": "2024-01-02 14:00", "exit_reason": "take_profit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_no_reentry_after_loss(entries, exits)
        assert out_entries["entry_id"].to_list() == [1]

    def test_win_then_loss_both_kept_but_reentry_blocked(self):
        """Win then loss on same day: both kept; re-entry after the loss is dropped."""
        engine = _make_engine()
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:30", "exit_reason": "take_profit"},
            {"entry_id": 2, "entry_time": "2024-01-02 10:30", "exit_reason": "stop_loss"},
            {"entry_id": 3, "entry_time": "2024-01-02 12:00", "exit_reason": "take_profit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_no_reentry_after_loss(entries, exits)
        assert sorted(out_entries["entry_id"].to_list()) == [1, 2]

    def test_multiple_days_with_losses(self):
        """Each loss only blocks the rest of its own day; other days are unaffected."""
        engine = _make_engine()
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:30", "exit_reason": "stop_loss"},
            {
                "entry_id": 2,
                "entry_time": "2024-01-02 11:00",
                "exit_reason": "take_profit",
            },  # blocked
            {
                "entry_id": 3,
                "entry_time": "2024-01-03 09:30",
                "exit_reason": "take_profit",
            },  # allowed
            {"entry_id": 4, "entry_time": "2024-01-04 09:30", "exit_reason": "stop_loss"},
            {
                "entry_id": 5,
                "entry_time": "2024-01-04 11:00",
                "exit_reason": "take_profit",
            },  # blocked
            {
                "entry_id": 6,
                "entry_time": "2024-01-05 09:30",
                "exit_reason": "take_profit",
            },  # allowed
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_no_reentry_after_loss(entries, exits)
        assert sorted(out_entries["entry_id"].to_list()) == [1, 3, 4, 6]

    def test_exits_filtered_consistently_with_entries(self):
        """Entries and exits that are dropped match — no orphaned exit rows."""
        engine = _make_engine()
        rows = [
            {"entry_id": 1, "entry_time": "2024-01-02 09:30", "exit_reason": "stop_loss"},
            {"entry_id": 2, "entry_time": "2024-01-02 11:00", "exit_reason": "take_profit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, out_exits = engine._enforce_no_reentry_after_loss(entries, exits)
        assert set(out_entries["entry_id"].to_list()) == set(out_exits["entry_id"].to_list())

    def test_empty_entries_returns_empty(self):
        """Empty inputs don't raise and return empty DataFrames."""
        engine = _make_engine()
        entries = pl.DataFrame(
            {
                "entry_id": pl.Series([], dtype=pl.Int64),
                "entry_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            }
        )
        exits = pl.DataFrame(
            {
                "entry_id": pl.Series([], dtype=pl.Int64),
                "exit_reason": pl.Series([], dtype=pl.String),
            }
        )
        out_entries, out_exits = engine._enforce_no_reentry_after_loss(entries, exits)
        assert out_entries.is_empty()
        assert out_exits.is_empty()

    def test_definition_field_defaults_to_false(self):
        """EntryConfig.no_reentry_after_loss defaults to False."""
        cfg = EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0)))
        assert cfg.no_reentry_after_loss is False

    def test_definition_field_accepts_true(self):
        """EntryConfig.no_reentry_after_loss can be set to True."""
        cfg = EntryConfig(
            window=EntryWindowConfig(start=time(9, 30), end=time(16, 0)),
            no_reentry_after_loss=True,
        )
        assert cfg.no_reentry_after_loss is True
