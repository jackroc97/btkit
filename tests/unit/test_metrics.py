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


# ---------------------------------------------------------------------------
# Bootstrap CI for mean P&L
# ---------------------------------------------------------------------------


class TestBootstrapMeanCI:
    def test_ci_contains_true_mean(self):
        """With 1000 identical values, the CI should tightly bracket the mean."""
        import numpy as np

        pnl = np.full(200, 50.0)
        lo, hi = PostProcessor._bootstrap_mean_ci(pnl, n_boot=2000, rng=np.random.default_rng(0))
        assert lo <= 50.0 <= hi

    def test_ci_width_shrinks_with_more_data(self):
        """Wider sample → narrower CI (law of large numbers)."""
        import numpy as np

        rng = np.random.default_rng(42)
        small = rng.normal(0, 10, size=20)
        large = rng.normal(0, 10, size=500)
        lo_s, hi_s = PostProcessor._bootstrap_mean_ci(
            small, n_boot=5000, rng=np.random.default_rng(0)
        )
        lo_l, hi_l = PostProcessor._bootstrap_mean_ci(
            large, n_boot=5000, rng=np.random.default_rng(0)
        )
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_empty_returns_zeros(self):
        import numpy as np

        lo, hi = PostProcessor._bootstrap_mean_ci(np.array([]))
        assert lo == 0.0 and hi == 0.0

    def test_seed_reproducible(self):
        import numpy as np

        pnl = np.random.default_rng(7).normal(100, 20, size=100)
        r1 = PostProcessor._bootstrap_mean_ci(pnl, n_boot=1000, rng=np.random.default_rng(99))
        r2 = PostProcessor._bootstrap_mean_ci(pnl, n_boot=1000, rng=np.random.default_rng(99))
        assert r1 == r2

    def test_metrics_returns_ci_keys(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 100.0},
                {
                    "open_mark": 5.0,
                    "exit_mark": 6.0,
                    "worst_mark": 6.0,
                    "net_pnl": -50.0,
                    "exit_reason": "stop_loss",
                },
            ],
        )
        m = PostProcessor(db, backtest_id=bid).metrics(seed=0)
        assert "mean_pnl_ci_lower" in m
        assert "mean_pnl_ci_upper" in m
        assert m["mean_pnl_ci_lower"] <= m["mean_pnl_ci_upper"]


# ---------------------------------------------------------------------------
# Wilson CI for win rate
# ---------------------------------------------------------------------------


class TestWilsonWinRateCI:
    def test_known_interval(self):
        """50 wins out of 100 trades — Wilson CI should be roughly [0.40, 0.60]."""
        lo, hi = PostProcessor._wilson_win_rate_ci(50, 100)
        assert pytest.approx(lo, abs=0.01) == 0.4013
        assert pytest.approx(hi, abs=0.01) == 0.5987

    def test_bounds_in_zero_one(self):
        """Extreme proportions must stay in [0, 1]."""
        lo_low, hi_low = PostProcessor._wilson_win_rate_ci(0, 10)
        lo_high, hi_high = PostProcessor._wilson_win_rate_ci(10, 10)
        assert lo_low >= 0.0
        assert hi_high <= 1.0

    def test_zero_total_returns_zeros(self):
        assert PostProcessor._wilson_win_rate_ci(0, 0) == (0.0, 0.0)

    def test_lo_le_hi(self):
        for n_wins, n_total in [(3, 10), (1, 5), (99, 100), (0, 50)]:
            lo, hi = PostProcessor._wilson_win_rate_ci(n_wins, n_total)
            assert lo <= hi

    def test_metrics_returns_win_rate_ci_keys(self, output_db):
        db, bid = _make_db_with_positions(
            output_db,
            [
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 100.0},
                {"open_mark": 5.0, "exit_mark": 4.0, "worst_mark": 5.5, "net_pnl": 100.0},
                {
                    "open_mark": 5.0,
                    "exit_mark": 6.0,
                    "worst_mark": 6.0,
                    "net_pnl": -50.0,
                    "exit_reason": "stop_loss",
                },
            ],
        )
        m = PostProcessor(db, backtest_id=bid).metrics(seed=0)
        assert "win_rate_ci_lower" in m
        assert "win_rate_ci_upper" in m
        assert 0.0 <= m["win_rate_ci_lower"] <= m["win_rate_ci_upper"] <= 1.0
