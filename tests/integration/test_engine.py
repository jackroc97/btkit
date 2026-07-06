"""
Integration tests for BacktestEngine — full single-run pipeline against the
fixture database. Each test loads a strategy YAML, runs the engine, and
verifies structural and semantic correctness of the output.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

from btkit.backtest.engine import BacktestEngine
from btkit.strategy.definition import (
    DeltaStep,
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    SimpleDeltaConfig,
    SteppedDeltaConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)
from btkit.strategy.loader import load_strategy

STRATEGIES_DIR = Path(__file__).parent.parent / "fixtures" / "strategies"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(strategy_name: str, input_db, output_db) -> int:
    strat = load_strategy(STRATEGIES_DIR / f"{strategy_name}.yaml")
    engine = BacktestEngine(input_db, output_db, strat)
    return engine.run()


def _con(output_db):
    return output_db._con


# ---------------------------------------------------------------------------
# Structural correctness (all runs)
# ---------------------------------------------------------------------------


class TestEngineStructural:
    def test_short_put_spread_produces_positions(self, input_db, output_db):
        _run("short_put_spread", input_db, output_db)
        n = _con(output_db).execute("SELECT COUNT(*) FROM position").fetchone()[0]
        assert n > 0

    def test_positions_have_two_legs_in_spread(self, input_db, output_db):
        _run("short_put_spread", input_db, output_db)
        bad = (
            _con(output_db)
            .execute(
                "SELECT COUNT(*) FROM ("
                "  SELECT position_id, COUNT(*) AS c FROM position_leg "
                "  GROUP BY position_id HAVING c != 2"
                ")"
            )
            .fetchone()[0]
        )
        assert bad == 0, "All spread positions should have exactly 2 legs"

    def test_no_temporal_violations(self, input_db, output_db):
        _run("short_put_spread", input_db, output_db)
        bad = (
            _con(output_db)
            .execute(
                "SELECT COUNT(*) FROM position "
                "WHERE exit_time IS NOT NULL AND exit_time <= open_time"
            )
            .fetchone()[0]
        )
        assert bad == 0

    def test_referential_integrity(self, input_db, output_db):
        _run("short_put_spread", input_db, output_db)
        orphans = (
            _con(output_db)
            .execute(
                "SELECT COUNT(*) FROM position_leg pl "
                "LEFT JOIN position p ON pl.position_id = p.id WHERE p.id IS NULL"
            )
            .fetchone()[0]
        )
        assert orphans == 0

    def test_valid_exit_reasons_only(self, input_db, output_db):
        _run("short_put_spread", input_db, output_db)
        valid = "('take_profit','stop_loss','condition','dte_exit','expiry')"
        bad = (
            _con(output_db)
            .execute(
                f"SELECT COUNT(*) FROM position "
                f"WHERE exit_reason NOT IN {valid} "
                f"AND exit_reason IS NOT NULL"
            )
            .fetchone()[0]
        )
        assert bad == 0

    def test_pnl_formula_correct(self, input_db, output_db):
        _run("short_put_spread", input_db, output_db)
        pnl_check = (
            "ABS(p.net_pnl - "
            "((p.open_mark - p.exit_mark)*x.m - p.slippage_cost - p.fee_cost)) > 0.01"
        )
        bad = (
            _con(output_db)
            .execute(
                "SELECT COUNT(*) FROM position p "
                "JOIN (SELECT position_id, MAX(multiplier) m FROM position_leg GROUP BY 1) x "
                f"ON x.position_id = p.id WHERE {pnl_check} AND p.exit_mark IS NOT NULL"
            )
            .fetchone()[0]
        )
        assert bad == 0

    def test_entry_greeks_populated(self, input_db, output_db):
        _run("short_put_spread", input_db, output_db)
        null_delta = (
            _con(output_db)
            .execute("SELECT COUNT(*) FROM position_leg WHERE entry_delta IS NULL")
            .fetchone()[0]
        )
        assert null_delta == 0, "entry_delta should be populated for all legs"


# ---------------------------------------------------------------------------
# Exit reason coverage
# ---------------------------------------------------------------------------


class TestExitReasons:
    def test_take_profit_produces_tp_exits(self, input_db, output_db):
        _run("exit_tp", input_db, output_db)
        reasons = {
            r[0]
            for r in _con(output_db).execute("SELECT DISTINCT exit_reason FROM position").fetchall()
        }
        assert "take_profit" in reasons or "expiry" in reasons

    def test_stop_loss_produces_sl_exits(self, input_db, output_db):
        _run("exit_sl", input_db, output_db)
        reasons = {
            r[0]
            for r in _con(output_db).execute("SELECT DISTINCT exit_reason FROM position").fetchall()
        }
        assert "stop_loss" in reasons or "expiry" in reasons

    def test_exit_condition_all_condition_exits(self, input_db, output_db):
        _run("exit_condition", input_db, output_db)
        reasons = {
            r[0]
            for r in _con(output_db).execute("SELECT DISTINCT exit_reason FROM position").fetchall()
        }
        assert reasons == {"condition"}

    def test_dte_exit_produces_dte_exits(self, input_db, output_db):
        _run("exit_dte", input_db, output_db)
        reasons = {
            r[0]
            for r in _con(output_db).execute("SELECT DISTINCT exit_reason FROM position").fetchall()
        }
        assert "dte_exit" in reasons or "expiry" in reasons

    def test_expiry_exit_all_expiry(self, input_db, output_db):
        _run("exit_expiry", input_db, output_db)
        reasons = {
            r[0]
            for r in _con(output_db).execute("SELECT DISTINCT exit_reason FROM position").fetchall()
        }
        assert reasons == {"expiry"}

    def test_expiry_exit_marks_non_negative(self, input_db, output_db):
        """
        Credit-spread expiry marks must be >= 0.  A negative exit_mark is
        physically impossible (lower-strike option cannot be worth less than
        the higher-strike option of the same type/expiry) and indicates a
        stale forward-fill contaminating the mark.  Settlement-based expiry
        marks are computed from underlying intrinsic values and are always >= 0.
        """
        _run("exit_expiry", input_db, output_db)
        bad = (
            _con(output_db)
            .execute("SELECT COUNT(*) FROM position WHERE exit_reason = 'expiry' AND exit_mark < 0")
            .fetchone()[0]
        )
        assert bad == 0, f"{bad} expiry exits have negative exit_mark"


# ---------------------------------------------------------------------------
# Indicator conditions
# ---------------------------------------------------------------------------


class TestIndicatorConditions:
    def test_indicator_entry_gate_reduces_count(self, input_db, output_db, tmp_path):
        """Indicator entry condition filters fewer entries than no-condition baseline."""
        from btkit.db.output_db import OutputDatabase

        # Baseline
        base_db = OutputDatabase(str(tmp_path / "base.db"))
        base_db.create_schema()
        _run("short_put_spread", input_db, base_db)
        n_base = base_db._con.execute("SELECT COUNT(*) FROM position").fetchone()[0]
        base_db.close()

        # Selective (sma_5 > sma_20)
        _run("indicator_conditions", input_db, output_db)
        n_sel = output_db._con.execute("SELECT COUNT(*) FROM position").fetchone()[0]

        assert n_base > 0
        assert 0 <= n_sel < n_base

    def test_indicator_exit_all_condition(self, input_db, output_db):
        _run("exit_condition", input_db, output_db)
        reasons = {
            r[0]
            for r in output_db._con.execute("SELECT DISTINCT exit_reason FROM position").fetchall()
        }
        assert reasons == {"condition"}


# ---------------------------------------------------------------------------
# Multi-trade (iron condor)
# ---------------------------------------------------------------------------


class TestMultiTrade:
    def test_iron_condor_two_trades(self, input_db, output_db):
        _run("iron_condor", input_db, output_db)
        trade_names = {
            r[0]
            for r in output_db._con.execute("SELECT DISTINCT trade_name FROM position").fetchall()
        }
        assert len(trade_names) == 2

    def test_iron_condor_four_legs(self, input_db, output_db):
        _run("iron_condor", input_db, output_db)
        bad = output_db._con.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT position_id, COUNT(*) AS c FROM position_leg GROUP BY 1 HAVING c != 2"
            ")"
        ).fetchone()[0]
        assert bad == 0, "Each condor wing should have exactly 2 legs"

    def test_one_at_a_time_per_trade(self, input_db, output_db):
        """No two positions of the same trade should overlap in time."""
        _run("iron_condor", input_db, output_db)
        # For each trade, check no open_time falls within another position's window
        for trade_name in ["put_spread", "call_spread"]:
            rows = output_db._con.execute(
                "SELECT open_time, exit_time FROM position WHERE trade_name = ? ORDER BY open_time",
                [trade_name],
            ).fetchall()
            for i in range(len(rows) - 1):
                prev_exit = rows[i][1]
                next_open = rows[i + 1][0]
                if prev_exit and next_open:
                    assert next_open >= prev_exit, (
                        f"Overlap in {trade_name}: position {i} exits at {prev_exit} "
                        f"but position {i + 1} opens at {next_open}"
                    )


# ---------------------------------------------------------------------------
# Stepped delta
# ---------------------------------------------------------------------------


def _make_stepped_strategy(steps: list[DeltaStep]) -> StrategyDefinition:
    """Strategy with a single short-put delta leg driven by sma_5 steps; no conditions."""
    return StrategyDefinition(
        name="stepped_delta_test",
        universe=UniverseConfig(
            start_date=date(2026, 4, 22),
            end_date=date(2026, 5, 21),
        ),
        trades=[
            TradeDefinition(
                name="short_put",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(
                    window=EntryWindowConfig(start=time(10, 0), end=time(12, 0)),
                ),
                legs=[
                    LegConfig(
                        name="sp",
                        right="put",
                        action="sell_to_open",
                        dte=21,
                        delta=SteppedDeltaConfig(
                            step_source="sma_5",
                            tolerance=0.15,
                            steps=steps,
                        ),
                    )
                ],
                exit=ExitConfig(stop_loss=2.0, take_profit=1.0),
            )
        ],
    )


class TestSteppedDelta:
    """
    Regression and correctness tests for IV-stepped delta leg selection.

    Uses the sma_5 indicator (5-bar SMA of ES close) which ranges from
    ~7150 to ~7540 during the test universe window.
    """

    def test_no_crash_when_no_conditions(self, input_db, output_db):
        """
        Regression: _load_indicators_once() must fetch indicators for stepped-delta
        legs even when entry.conditions and exit.conditions are both empty.
        Before the fix this raised ColumnNotFoundError inside _select_legs.
        """
        strat = _make_stepped_strategy(
            steps=[DeltaStep(target=-0.25, tolerance=0.10)]  # catch-all only
        )
        engine = BacktestEngine(input_db, output_db, strat)
        engine.run()
        n = output_db._con.execute("SELECT COUNT(*) FROM position").fetchone()[0]
        assert n > 0, "Should produce at least one position with a catch-all step"

    def test_catch_all_matches_flat_delta(self, input_db, output_db):
        """
        A catch-all stepped config (single step, no below) with the same target and
        tolerance as a flat SimpleDeltaConfig should produce identical positions.
        """
        stepped_strat = _make_stepped_strategy(steps=[DeltaStep(target=-0.25, tolerance=0.10)])
        flat_strat = StrategyDefinition(
            name="flat_delta_test",
            universe=stepped_strat.universe,
            trades=[
                TradeDefinition(
                    name="short_put",
                    instrument=stepped_strat.trades[0].instrument,
                    entry=stepped_strat.trades[0].entry,
                    legs=[
                        LegConfig(
                            name="sp",
                            right="put",
                            action="sell_to_open",
                            dte=21,
                            delta=SimpleDeltaConfig(target=-0.25, tolerance=0.10),
                        )
                    ],
                    exit=stepped_strat.trades[0].exit,
                )
            ],
        )

        import pathlib
        import tempfile

        from btkit.db.output_db import OutputDatabase

        with tempfile.TemporaryDirectory() as tmp:
            flat_out = OutputDatabase(str(pathlib.Path(tmp) / "flat.db"))
            flat_out.create_schema()
            BacktestEngine(input_db, flat_out, flat_strat).run()
            BacktestEngine(input_db, output_db, stepped_strat).run()

            n_flat = flat_out._con.execute("SELECT COUNT(*) FROM position").fetchone()[0]
            n_stepped = output_db._con.execute("SELECT COUNT(*) FROM position").fetchone()[0]

        assert n_stepped == n_flat, (
            f"Catch-all stepped config produced {n_stepped} positions vs {n_flat} for flat delta"
        )

    def test_two_steps_both_fire(self, input_db, output_db):
        """
        With a threshold of 7300 the sma_5 values during the entry window split
        roughly evenly (below: ~1200 bars, above: ~1440 bars), so both steps
        should produce positions.

        Step 1 (sma_5 < 7300): target -0.10  — shallower delta
        Catch-all (sma_5 >= 7300): target -0.25 — deeper delta

        Verify that entry_delta values from both steps appear in the results.
        The midpoint -0.175 separates the two targets; both sides should be populated.
        """
        strat = _make_stepped_strategy(
            steps=[
                DeltaStep(below=7300.0, target=-0.10, tolerance=0.15),
                DeltaStep(target=-0.25, tolerance=0.15),  # catch-all
            ]
        )
        engine = BacktestEngine(input_db, output_db, strat)
        engine.run()

        deltas = [
            row[0]
            for row in output_db._con.execute("SELECT entry_delta FROM position_leg").fetchall()
            if row[0] is not None
        ]
        assert len(deltas) > 0, "No positions with delta data"

        midpoint = -0.175
        near_shallow = sum(1 for d in deltas if d > midpoint)  # closer to -0.10
        near_deep = sum(1 for d in deltas if d <= midpoint)  # closer to -0.25

        assert near_shallow > 0, (
            "No positions from step 1 (sma_5 < 7300, target -0.10). "
            f"All {len(deltas)} deltas: {sorted(deltas)[:5]}..."
        )
        assert near_deep > 0, (
            "No positions from catch-all (sma_5 >= 7300, target -0.25). "
            f"All {len(deltas)} deltas: {sorted(deltas)[:5]}..."
        )


# ---------------------------------------------------------------------------
# Forward-contract visibility (quarterly roll gap)
# ---------------------------------------------------------------------------


class TestForwardContractVisibility:
    """
    Regression for the bug where options listed under the *next* quarterly
    futures contract were invisible when target DTE exceeded the front-month's
    remaining life.

    Test DB facts used here:
      - ESM6 (id=42140864) expires 2026-06-18; its option chain ends 2026-06-17.
      - ESU6 (id=42140870) option chain starts 2026-06-18.
      - For a 45-DTE ± 10 strategy, ESM6 has qualifying options only through
        ~2026-05-13. From 2026-05-14 onward ALL 35-55 DTE options are on ESU6.
      - Before the fix, entries were absent for 2026-05-14 to 2026-05-21
        because ts_event_underlying contained only ESM6 pairs.
    """

    def _make_45dte_strategy(self) -> StrategyDefinition:
        return StrategyDefinition(
            name="forward_contract_test",
            universe=UniverseConfig(
                start_date=date(2026, 4, 22),
                end_date=date(2026, 5, 21),
            ),
            trades=[
                TradeDefinition(
                    name="short_put",
                    instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                    entry=EntryConfig(
                        window=EntryWindowConfig(start=time(9, 30), end=time(15, 0)),
                    ),
                    legs=[
                        LegConfig(
                            name="sp",
                            right="put",
                            action="sell_to_open",
                            dte=45,
                            dte_tolerance=10,
                            delta=SimpleDeltaConfig(target=-0.25, tolerance=0.20),
                        )
                    ],
                    exit=ExitConfig(stop_loss=3.0, take_profit=0.5),
                )
            ],
        )

    def test_entries_found_during_next_contract_window(self, input_db, output_db):
        """
        Entries must be produced during 2026-05-14 to 2026-05-21 — the period
        where all 35-55 DTE puts are under ESU6, not ESM6.  Before the fix,
        zero positions opened in this window.
        """
        strat = self._make_45dte_strategy()
        engine = BacktestEngine(input_db, output_db, strat)
        engine.run()

        positions = output_db._con.execute(
            "SELECT COUNT(*) FROM position "
            "WHERE open_time >= '2026-05-14' AND open_time < '2026-05-22'"
        ).fetchone()[0]

        assert positions > 0, (
            "No positions opened during 2026-05-14 to 2026-05-21. "
            "The forward-contract visibility fix may be broken: options on ESU6 "
            "are invisible when ts_event_underlying only contains ESM6 pairs."
        )

    def test_entries_span_full_period(self, input_db, output_db):
        """
        Positions should be found across the whole test window, not just the
        ESM6-only period.  Total count with fix should exceed that of a strategy
        artificially restricted to ESM6-covered dates.
        """

        strat = self._make_45dte_strategy()
        engine = BacktestEngine(input_db, output_db, strat)
        engine.run()

        n_early = output_db._con.execute(
            "SELECT COUNT(*) FROM position "
            "WHERE open_time >= '2026-04-22' AND open_time < '2026-05-14'"
        ).fetchone()[0]
        n_late = output_db._con.execute(
            "SELECT COUNT(*) FROM position "
            "WHERE open_time >= '2026-05-14' AND open_time < '2026-05-22'"
        ).fetchone()[0]

        assert n_early > 0, "No positions in ESM6-only window (expected entries here regardless)"
        assert n_late > 0, "No positions in ESU6 window (forward-contract visibility fix required)"
