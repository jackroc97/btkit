"""
Unit tests for the roll block feature.

Covers:
  - RollConfig validation (at_least_one_trigger)
  - roll field default on TradeDefinition
  - _enforce_max_entries_per_day roll re-entry bypass
  - roll + no_reentry_after_loss interaction (roll does NOT block re-entry)
  - roll.conditions — condition-only trigger, combined trigger, _need_vega activation
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
                    LegConfig(name="short", right="put", action="sell_to_open", dte=0, delta={"target": -0.16}),
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
                LegConfig(name="short", right="put", action="sell_to_open", dte=0, delta={"target": -0.16}),
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
                LegConfig(name="short", right="put", action="sell_to_open", dte=0, delta={"target": -0.16}),
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

    def test_roll_reentry_tp_same_day_allows_subsequent_regular_entry(self):
        """
        Regression: original position opened on a prior day, rolled on the
        current day.  The roll re-entry TPs same-day.  A subsequent regular
        candidate on the same day must survive as rank-1 (first non-exempt).

        Bug: cum_count().over("_entry_date") counted the roll re-entry in
        the per-day total, so the regular candidate got rank=2 → blocked.
        Fix: (~_is_roll_reentry).cast(Int32).cum_sum() excludes exempt rows.
        """
        engine = _make_engine(max_entries_per_day=1)
        rows = [
            # Entry 1 opened Jan 2, exits via roll on Jan 3 10:00
            {"entry_id": 1, "entry_time": "2024-01-02 09:45", "exit_time": "2024-01-03 10:00", "exit_reason": "roll"},
            # Entry 2: roll re-entry Jan 3 10:01 (exempt), TPs at 11:00
            {"entry_id": 2, "entry_time": "2024-01-03 10:01", "exit_time": "2024-01-03 11:00", "exit_reason": "take_profit"},
            # Entry 3: first non-exempt entry on Jan 3 after TP — rank should be 1
            {"entry_id": 3, "entry_time": "2024-01-03 11:01", "exit_time": "2024-01-03 15:00", "exit_reason": "dte_exit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_max_entries_per_day(
            entries, exits, max_entries=1, roll_exit_ids={1}
        )
        # All three must survive: entry 1 (Jan 2, rank=1), entry 2 (exempt),
        # entry 3 (Jan 3 first non-exempt, rank=1).
        assert sorted(out_entries["entry_id"].to_list()) == [1, 2, 3]

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


class TestRollConditions:
    """
    Tests for roll.conditions — expression-based roll triggers that share the
    exit condition namespace.
    """

    def test_conditions_only_is_valid(self):
        """conditions alone (no dte or vega) is a valid trigger."""
        cfg = RollConfig(conditions=["position_mark - open_mark >= 10.0"])
        assert cfg.conditions == ["position_mark - open_mark >= 10.0"]
        assert cfg.dte is None
        assert cfg.vega is None

    def test_empty_conditions_with_no_other_triggers_raises(self):
        """Empty conditions list with no dte/vega still raises."""
        with pytest.raises(Exception, match="roll requires at least one trigger"):
            RollConfig(conditions=[])

    def test_conditions_combined_with_dte(self):
        cfg = RollConfig(dte=10, conditions=["position_mark - open_mark >= 10.0"])
        assert cfg.dte == 10
        assert len(cfg.conditions) == 1

    def test_multiple_conditions(self):
        cfg = RollConfig(conditions=[
            "position_mark - open_mark >= 10.0",
            "_spread_vega < 0.3 * open_vega",
        ])
        assert len(cfg.conditions) == 2

    def test_need_vega_triggered_by_spread_vega_in_roll_condition(self):
        """_need_vega activates when a roll condition references _spread_vega."""
        from btkit.strategy.definition import ExitConfig
        roll = RollConfig(conditions=["_spread_vega < 0.5"])
        exit_cfg = ExitConfig()
        result = (
            exit_cfg.vega_exit is not None
            or roll.vega is not None
            or any("_spread_vega" in c or "open_vega" in c for c in exit_cfg.conditions)
            or any("_spread_vega" in c or "open_vega" in c for c in roll.conditions)
        )
        assert result is True

    def test_need_vega_triggered_by_open_vega_in_roll_condition(self):
        """_need_vega activates when a roll condition references open_vega."""
        from btkit.strategy.definition import ExitConfig
        roll = RollConfig(conditions=["_spread_vega < 0.3 * open_vega"])
        exit_cfg = ExitConfig()
        result = (
            exit_cfg.vega_exit is not None
            or roll.vega is not None
            or any("_spread_vega" in c or "open_vega" in c for c in exit_cfg.conditions)
            or any("_spread_vega" in c or "open_vega" in c for c in roll.conditions)
        )
        assert result is True

    def test_need_vega_not_triggered_by_unrelated_roll_condition(self):
        """_need_vega stays False when roll conditions don't reference vega columns."""
        from btkit.strategy.definition import ExitConfig
        roll = RollConfig(conditions=["position_mark - open_mark >= 10.0"])
        exit_cfg = ExitConfig()
        result = (
            exit_cfg.vega_exit is not None
            or roll.vega is not None
            or any("_spread_vega" in c or "open_vega" in c for c in exit_cfg.conditions)
            or any("_spread_vega" in c or "open_vega" in c for c in roll.conditions)
        )
        assert result is False

    def test_roll_trigger_fires_on_condition(self):
        """
        Smoke-test: a roll condition expression is correctly parsed and ORed
        into the roll_trigger Polars expression without raising.
        """
        from btkit.strategy.loader import parse_condition
        import polars as pl

        roll_cfg = RollConfig(conditions=["position_mark - open_mark >= 10.0"])

        m = pl.DataFrame({
            "position_mark": [5.0, 15.0, 8.0],
            "open_mark":     [0.0,  0.0,  0.0],
            "_local_sec":    [36900, 36900, 36900],
        })

        roll_trigger = pl.lit(False)
        for cond_str in roll_cfg.conditions:
            roll_trigger = roll_trigger | parse_condition(cond_str)

        result = m.with_columns(roll_trigger.alias("_roll"))["_roll"].to_list()
        assert result == [False, True, False]

    def test_roll_condition_combined_with_dte_uses_or_logic(self):
        """Either the dte trigger OR the condition trigger should fire the roll."""
        from btkit.strategy.loader import parse_condition
        import polars as pl

        roll_cfg = RollConfig(dte=5, conditions=["position_mark - open_mark >= 10.0"])

        m = pl.DataFrame({
            "position_mark": [15.0, 3.0,  3.0],
            "open_mark":     [ 0.0, 0.0,  0.0],
            "_dte_now":      [  20,  20,    3],
        })

        roll_trigger = pl.lit(False)
        if roll_cfg.dte is not None:
            roll_trigger = roll_trigger | (pl.col("_dte_now") <= pl.lit(int(roll_cfg.dte)))
        for cond_str in roll_cfg.conditions:
            roll_trigger = roll_trigger | parse_condition(cond_str)

        result = m.with_columns(roll_trigger.alias("_roll"))["_roll"].to_list()
        # row 0: condition fires (15 >= 10); row 1: neither; row 2: dte fires (3 <= 5)
        assert result == [True, False, True]


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


# ---------------------------------------------------------------------------
# _enforce_entries ghost-gate regression tests
# ---------------------------------------------------------------------------

class TestEnforceEntriesGhostGate:
    """
    Regression tests for the ghost-gate bug.

    When _enforce_one_at_a_time and _enforce_max_entries_per_day ran as two
    separate sequential passes, the first pass could accept a "ghost" candidate
    at or just after the prior exit timestamp (via the `>=` gate check) and
    advance the gate to that ghost's exit time.  The second pass would then
    remove the ghost (daily cap exceeded), but the gate remained at the ghost's
    distant exit time, blocking valid entries for weeks.

    _enforce_entries applies both constraints in a single walk so the gate only
    advances when an entry survives both checks.
    """

    def test_ghost_at_exact_exit_time_does_not_advance_gate(self):
        """
        Gap C pattern: E1 opens, TPs at 14:29.  Ghost E2b enters at exactly
        14:29 (same timestamp as E1's exit, allowed by >= gate).  With
        max_entries_per_day=1, E2b is rank=2 and must be dropped.  The gate
        should stay at E1's exit (14:29), so Jan 9 entries are accepted.
        """
        engine = _make_engine(max_entries_per_day=1)
        rows = [
            # E1: opens Jan 8 09:46, TPs 14:29
            {"entry_id": 1, "entry_time": "2024-01-08 09:46", "exit_time": "2024-01-08 14:29", "exit_reason": "take_profit"},
            # E2b: ghost at exact TP time — accepted by >= gate but rank=2 for Jan 8
            {"entry_id": 2, "entry_time": "2024-01-08 14:29", "exit_time": "2024-02-20 12:00", "exit_reason": "dte_exit"},
            # E3: first valid candidate next day — should be accepted
            {"entry_id": 3, "entry_time": "2024-01-09 09:46", "exit_time": "2024-02-23 10:00", "exit_reason": "dte_exit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_entries(entries, exits, max_entries_per_day=1)
        # E1 and E3 must survive; E2b is blocked by the daily cap.
        assert sorted(out_entries["entry_id"].to_list()) == [1, 3]

    def test_ghost_chain_after_exit_does_not_advance_gate(self):
        """
        Gap A pattern: E2 opens and TPs at 10:38.  Multiple ghost candidates
        at 10:41, 10:42 (all rank=2+ for the same day) must not advance the
        gate.  May 6 entry should be accepted.
        """
        engine = _make_engine(max_entries_per_day=1)
        rows = [
            # E2 opens May 5 09:55, TPs 10:38
            {"entry_id": 2, "entry_time": "2022-05-05 09:55", "exit_time": "2022-05-05 10:38", "exit_reason": "take_profit"},
            # Ghost candidates on May 5 after the TP — all rank≥2
            {"entry_id": 3, "entry_time": "2022-05-05 10:41", "exit_time": "2022-05-05 10:42", "exit_reason": "take_profit"},
            {"entry_id": 4, "entry_time": "2022-05-05 10:42", "exit_time": "2022-05-05 10:43", "exit_reason": "take_profit"},
            {"entry_id": 5, "entry_time": "2022-05-05 10:44", "exit_time": "2022-06-14 09:45", "exit_reason": "dte_exit"},
            # Next-day candidate — should not be blocked by ghost gates
            {"entry_id": 6, "entry_time": "2022-05-06 09:45", "exit_time": "2022-06-19 10:00", "exit_reason": "dte_exit"},
        ]
        entries, exits = _make_frames(rows)
        out_entries, _ = engine._enforce_entries(entries, exits, max_entries_per_day=1)
        # E2 and E6 survive; ghosts 3, 4, 5 are daily-cap blocked.
        assert sorted(out_entries["entry_id"].to_list()) == [2, 6]

    def test_null_exit_ghost_does_not_advance_gate(self):
        """
        A candidate with NULL exit_time (gate_time=0 after fill_null) and
        max_entries_per_day=1 should be accepted (rank=1), then subsequent
        candidates on the same day are blocked by daily cap — NOT by a
        phantom gate — and the next-day entry fires normally.
        """
        import zoneinfo
        engine = _make_engine(max_entries_per_day=1)
        # Build entries with one null exit_time
        entries = pl.DataFrame({
            "entry_id":   pl.Series([1, 2, 3], dtype=pl.Int64),
            "entry_time": pl.Series([
                _ts("2024-01-05 12:15"),
                _ts("2024-01-05 12:16"),
                _ts("2024-01-08 09:46"),
            ]).dt.cast_time_unit("us"),
        })
        exits = pl.DataFrame({
            "entry_id":   pl.Series([1, 2, 3], dtype=pl.Int64),
            "exit_reason": pl.Series(["dte_exit", "dte_exit", "dte_exit"], dtype=pl.String),
            "exit_time":  pl.Series([None, _ts("2024-02-09 09:45"), _ts("2024-01-08 14:29")],
                                     dtype=pl.Datetime("us", "UTC")),
        })
        out_entries, _ = engine._enforce_entries(entries, exits, max_entries_per_day=1)
        # Entry 1 (null exit): accepted, gate=0.
        # Entry 2: gate=0 (clear), but daily cap for Jan 5 = 1 already hit → BLOCKED.
        # Entry 3 (Jan 8): gate=0, fresh day → accepted.
        assert sorted(out_entries["entry_id"].to_list()) == [1, 3]

    def test_roll_reentry_exempt_in_joint_walk(self):
        """
        Roll re-entry is exempt from the daily cap in the joint walk, exactly
        as in _enforce_max_entries_per_day.
        """
        engine = _make_engine(max_entries_per_day=1)
        rows = [
            # Original position opens Feb 1, rolls Feb 3 09:49
            {"entry_id": 1, "entry_time": "2025-02-01 12:45", "exit_time": "2025-02-03 09:49", "exit_reason": "roll"},
            # Roll re-entry: first entry on Feb 3 after 09:49 — exempt from cap
            {"entry_id": 2, "entry_time": "2025-02-03 09:54", "exit_time": "2025-02-03 10:25", "exit_reason": "take_profit"},
            # Regular candidate on Feb 3 after the TP — rank=1 (first non-exempt)
            {"entry_id": 3, "entry_time": "2025-02-03 10:26", "exit_time": "2025-03-03 11:00", "exit_reason": "roll"},
        ]
        entries, exits = _make_frames(rows)
        # roll_exit_ids = {1} (original position rolls)
        out_entries, _ = engine._enforce_entries(entries, exits, max_entries_per_day=1)
        # All three survive: 1 (original), 2 (exempt roll re-entry), 3 (rank=1 regular).
        assert sorted(out_entries["entry_id"].to_list()) == [1, 2, 3]
