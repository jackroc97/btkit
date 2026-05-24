"""
Unit tests for PostProcessor metrics.

Writes known positions directly into a fresh OutputDatabase and verifies that
every metric matches a manually computed expected value. No InputDatabase or
engine needed — PostProcessor only reads from the output DB.
"""

from __future__ import annotations

import pytest

from btkit.analysis.metrics import PostProcessor
from btkit.db.output_db import OutputDatabase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_backtest(db: OutputDatabase, initial_equity: float = 100_000.0) -> int:
    """Insert a minimal backtest record and return its id."""
    return db.write_backtest(
        {
            "strategy_name": "test",
            "strategy_version": "1.0",
            "strategy_params": {},
            "initial_equity": initial_equity,
            "slippage_pct": 0.0,
            "fee_per_contract": 0.0,
        }
    )


def _insert_position(
    db: OutputDatabase,
    backtest_id: int,
    *,
    open_mark: float,
    exit_mark: float,
    worst_mark: float,
    net_pnl: float,
    exit_reason: str = "take_profit",
    open_time: str = "2026-01-10 10:30:00+00",
    exit_time: str = "2026-01-10 14:00:00+00",
) -> None:
    db._con.execute(
        """
        INSERT INTO position
            (id, backtest_id, trade_name, open_time, exit_time, exit_reason,
             open_mark, exit_mark, worst_mark, slippage_cost, fee_cost, net_pnl)
        VALUES (nextval('seq_position'), ?, 'trade1', ?, ?, ?, ?, ?, ?, 0, 0, ?)
        """,
        [backtest_id, open_time, exit_time, exit_reason, open_mark, exit_mark, worst_mark, net_pnl],
    )


def _make_db_with_positions(
    output_db: OutputDatabase,
    positions: list[dict],
    initial_equity: float = 100_000.0,
) -> tuple[OutputDatabase, int]:
    """Seed a backtest and insert a list of position dicts."""
    # Need a sequence for position IDs
    output_db._con.execute("CREATE SEQUENCE IF NOT EXISTS seq_position START 1")
    bid = _seed_backtest(output_db, initial_equity)
    for p in positions:
        _insert_position(output_db, bid, **p)
    return output_db, bid


# ---------------------------------------------------------------------------
# Basic metric correctness
# ---------------------------------------------------------------------------


class TestMetricsBasic:
    def test_net_profit(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 50.0},
                {
                    "open_mark": 5.0,
                    "exit_mark": 4.0,
                    "worst_mark": 5.5,
                    "net_pnl": -30.0,
                    "exit_reason": "stop_loss",
                },
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        m = pp.metrics()
        assert m["net_profit"] == pytest.approx(20.0, abs=0.01)

    def test_total_trades(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 50.0},
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 50.0},
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 50.0},
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        assert pp.metrics()["total_trades"] == 3

    def test_percent_profitable(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 100.0},
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 100.0},
                {
                    "open_mark": 5.0,
                    "exit_mark": 7.0,
                    "worst_mark": 7.0,
                    "net_pnl": -100.0,
                    "exit_reason": "stop_loss",
                },
                {
                    "open_mark": 5.0,
                    "exit_mark": 7.0,
                    "worst_mark": 7.0,
                    "net_pnl": -100.0,
                    "exit_reason": "stop_loss",
                },
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        assert pp.metrics()["percent_profitable"] == pytest.approx(0.5)

    def test_profit_factor(self, output_db):
        # gross_wins=200, gross_losses=100 → PF=2.0
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 200.0},
                {
                    "open_mark": 5.0,
                    "exit_mark": 7.0,
                    "worst_mark": 7.0,
                    "net_pnl": -100.0,
                    "exit_reason": "stop_loss",
                },
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        assert pp.metrics()["profit_factor"] == pytest.approx(2.0)

    def test_profit_factor_no_losses_is_inf(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 100.0},
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        assert pp.metrics()["profit_factor"] == float("inf")

    def test_avg_win_avg_loss(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 100.0},
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 200.0},
                {
                    "open_mark": 5.0,
                    "exit_mark": 7.0,
                    "worst_mark": 7.0,
                    "net_pnl": -50.0,
                    "exit_reason": "stop_loss",
                },
                {
                    "open_mark": 5.0,
                    "exit_mark": 7.0,
                    "worst_mark": 7.0,
                    "net_pnl": -150.0,
                    "exit_reason": "stop_loss",
                },
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        m = pp.metrics()
        assert m["avg_win"] == pytest.approx(150.0, abs=0.01)
        assert m["avg_loss"] == pytest.approx(-100.0, abs=0.01)

    def test_median_pnl(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 10.0},
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 20.0},
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 30.0},
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        assert pp.metrics()["median_pnl"] == pytest.approx(20.0, abs=0.01)


# ---------------------------------------------------------------------------
# MAE metrics
# ---------------------------------------------------------------------------


class TestMAEMetrics:
    def test_avg_mae(self, output_db):
        # MAE = |worst_mark - open_mark|
        # pos1: |5.5 - 5.0| = 0.5, pos2: |6.0 - 4.0| = 2.0 → avg = 1.25
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 50.0},
                {"open_mark": 4.0, "exit_mark": 3.0, "worst_mark": 6.0, "net_pnl": 50.0},
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        m = pp.metrics()
        assert m["avg_mae"] == pytest.approx(1.25, abs=0.01)

    def test_worst_mae(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 50.0},
                {"open_mark": 4.0, "exit_mark": 3.0, "worst_mark": 10.0, "net_pnl": 50.0},
            ],
        )
        pp = PostProcessor(db, backtest_id=bid)
        m = pp.metrics()
        assert m["worst_mae"] == pytest.approx(6.0, abs=0.01)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class TestDrawdown:
    def test_no_drawdown_all_wins(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {
                    "open_mark": 5.0,
                    "exit_mark": 4.0,
                    "worst_mark": 5.5,
                    "net_pnl": 100.0,
                    "exit_time": "2026-01-10 14:00:00+00",
                },
                {
                    "open_mark": 5.0,
                    "exit_mark": 4.0,
                    "worst_mark": 5.5,
                    "net_pnl": 100.0,
                    "exit_time": "2026-01-11 14:00:00+00",
                },
            ],
            initial_equity=100_000.0,
        )
        pp = PostProcessor(db, backtest_id=bid)
        m = pp.metrics()
        assert m["max_drawdown"] == pytest.approx(0.0, abs=0.01)

    def test_drawdown_after_loss(self, output_db):
        # equity: 100k → 100.2k → 100.1k → 100.4k
        # peak=100.2k, dd=100.2k-100.1k=100
        db, bid = _make_db_with_positions(
            output_db,
            [
                {
                    "open_mark": 5.0,
                    "exit_mark": 4.0,
                    "worst_mark": 5.0,
                    "net_pnl": 200.0,
                    "exit_time": "2026-01-10 14:00:00+00",
                },
                {
                    "open_mark": 5.0,
                    "exit_mark": 7.0,
                    "worst_mark": 7.0,
                    "net_pnl": -100.0,
                    "exit_reason": "stop_loss",
                    "exit_time": "2026-01-11 14:00:00+00",
                },
                {
                    "open_mark": 5.0,
                    "exit_mark": 4.0,
                    "worst_mark": 5.0,
                    "net_pnl": 300.0,
                    "exit_time": "2026-01-12 14:00:00+00",
                },
            ],
            initial_equity=100_000.0,
        )
        pp = PostProcessor(db, backtest_id=bid)
        m = pp.metrics()
        assert m["max_drawdown"] == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestEmptyResults:
    def test_no_positions_returns_zeros(self, output_db):
        output_db._con.execute("CREATE SEQUENCE IF NOT EXISTS seq_position START 1")
        bid = _seed_backtest(output_db)
        pp = PostProcessor(output_db, backtest_id=bid)
        m = pp.metrics()
        assert m["total_trades"] == 0
        assert m["net_profit"] == 0.0

    def test_equity_curve_empty(self, output_db):
        output_db._con.execute("CREATE SEQUENCE IF NOT EXISTS seq_position START 1")
        bid = _seed_backtest(output_db)
        pp = PostProcessor(output_db, backtest_id=bid)
        ec = pp.equity_curve()
        assert ec.is_empty()
