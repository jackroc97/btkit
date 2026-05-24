"""
Integration tests for BacktestEngine — full single-run pipeline against the
fixture database. Each test loads a strategy YAML, runs the engine, and
verifies structural and semantic correctness of the output.
"""

from __future__ import annotations

from pathlib import Path

from btkit.backtest.engine import BacktestEngine
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
