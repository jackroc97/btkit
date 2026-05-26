"""
PostProcessor — loads backtest results and computes standard metrics.

Supports single-run (backtest_id) and study-run (study_id) analysis.
Single-run metrics and terminal output are fully implemented. Study-level
summary (study_summary()) returns per-combination metrics. Heatmap analysis
requires a StudyExpander.params_df and is deferred.

MAE (Maximum Adverse Excursion) is derived from worst_mark and open_mark stored
in the position table — no additional data collection needed.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from btkit.db.output_db import OutputDatabase


class PostProcessor:
    def __init__(
        self,
        output_db: OutputDatabase,
        backtest_id: int | None = None,
        study_id: int | None = None,
        matrix_id: int | None = None,  # deprecated alias for study_id
    ) -> None:
        """
        Initialise with at most one of backtest_id (single run) or study_id
        (all runs from a study). If neither is given, the most recent backtest
        is used. matrix_id is a deprecated alias for study_id.
        """
        effective_study_id = study_id or matrix_id
        if backtest_id is not None and effective_study_id is not None:
            raise ValueError("Provide at most one of backtest_id or study_id")
        self.output_db = output_db
        self.backtest_id = backtest_id
        self.study_id = effective_study_id

    def metrics(self) -> dict[str, Any]:
        """
        Compute and return standard backtest metrics.

        Returns:
            net_profit, total_trades, percent_profitable,
            profit_factor, avg_win, avg_loss, median_pnl,
            max_drawdown, max_drawdown_pct, cagr, mar,
            sharpe_ratio, sortino_ratio, calmar_ratio,
            premium_capture_rate,
            avg_mae, median_mae, worst_mae

        MAE metrics derived from: abs(worst_mark - open_mark) per position.
        Only valid when initialised with backtest_id (single run).
        """
        positions = self._load_positions()
        if positions.is_empty():
            return self._empty_metrics()

        net_pnl = positions["net_pnl"]
        wins = net_pnl.filter(net_pnl > 0)
        losses = net_pnl.filter(net_pnl < 0)

        total_trades = len(positions)
        net_profit = float(net_pnl.sum())
        n_wins = len(wins)
        percent_profitable = n_wins / total_trades if total_trades > 0 else 0.0

        wins_total = float(wins.sum()) if n_wins > 0 else 0.0
        losses_total = abs(float(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (wins_total / losses_total) if losses_total > 0 else float("inf")

        avg_win = float(wins.mean()) if n_wins > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        median_pnl = float(net_pnl.median())

        # MAE: Maximum Adverse Excursion = most adverse mark seen during the trade
        mae_series = (positions["worst_mark"] - positions["open_mark"]).abs()
        avg_mae = float(mae_series.mean())
        median_mae = float(mae_series.median())
        worst_mae = float(mae_series.max())

        # Premium capture rate: fraction of initial credit captured as profit
        gross_pnl = positions["open_mark"] - positions["exit_mark"]
        open_abs = positions["open_mark"].abs()
        cap_rate_vals = (gross_pnl / open_abs).filter(open_abs > 0)
        premium_capture_rate = float(cap_rate_vals.mean()) if not cap_rate_vals.is_empty() else 0.0

        # Equity curve and drawdown
        initial_equity = self._load_initial_equity()
        equity_df = self.equity_curve()
        max_dd, max_dd_pct = self._max_drawdown(equity_df, initial_equity)

        # CAGR
        first_open = positions["open_time"].min()
        last_exit = positions["exit_time"].max()
        years = 0.0
        if first_open is not None and last_exit is not None:
            delta_us = (
                (last_exit - first_open).total_seconds()
                if hasattr((last_exit - first_open), "total_seconds")
                else 0.0
            )
            years = delta_us / (365.25 * 24 * 3600)
        if years > 0 and initial_equity > 0:
            final_equity = initial_equity + net_profit
            cagr = (final_equity / initial_equity) ** (1.0 / years) - 1.0
        else:
            cagr = 0.0

        mar = (cagr / max_dd_pct) if max_dd_pct > 0 else float("inf")
        calmar_ratio = mar

        # Daily returns for Sharpe / Sortino (trade-exit based)
        daily = (
            positions.with_columns(pl.col("exit_time").dt.date().alias("exit_date"))
            .group_by("exit_date")
            .agg(pl.col("net_pnl").sum())
            .sort("exit_date")
        )["net_pnl"]

        mean_daily = float(daily.mean()) if not daily.is_empty() else 0.0
        std_daily = float(daily.std()) if len(daily) > 1 else 0.0

        ann_factor = math.sqrt(252)
        sharpe_ratio = (mean_daily / std_daily * ann_factor) if std_daily > 0 else 0.0

        downside = daily.filter(daily < 0)
        std_down = float(downside.std()) if len(downside) > 1 else 0.0
        sortino_ratio = (mean_daily / std_down * ann_factor) if std_down > 0 else float("inf")

        return {
            "net_profit": round(net_profit, 2),
            "total_trades": total_trades,
            "percent_profitable": round(percent_profitable, 4),
            "profit_factor": round(profit_factor, 4)
            if math.isfinite(profit_factor)
            else float("inf"),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "median_pnl": round(median_pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "cagr": round(cagr, 4),
            "mar": round(mar, 4) if math.isfinite(mar) else float("inf"),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "sortino_ratio": round(sortino_ratio, 4)
            if math.isfinite(sortino_ratio)
            else float("inf"),
            "calmar_ratio": round(calmar_ratio, 4) if math.isfinite(calmar_ratio) else float("inf"),
            "premium_capture_rate": round(premium_capture_rate, 4),
            "avg_mae": round(avg_mae, 2),
            "median_mae": round(median_mae, 2),
            "worst_mae": round(worst_mae, 2),
        }

    def equity_curve(self) -> pl.DataFrame:
        """
        Returns a DataFrame of cumulative equity over time.
        Columns: exit_time, equity (initial_equity + cumulative net_pnl at each exit).
        Rows are ordered by exit_time.
        """
        positions = self._load_positions()
        if positions.is_empty():
            return pl.DataFrame(
                {
                    "exit_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                    "equity": pl.Series([], dtype=pl.Float64),
                }
            )
        initial = self._load_initial_equity()
        return (
            positions.sort("exit_time")
            .with_columns((pl.lit(initial) + pl.col("net_pnl").cum_sum()).alias("equity"))
            .select(["exit_time", "equity"])
        )

    def trade_pnl_series(self) -> pl.DataFrame:
        """
        Returns one row per trade with columns:
            trade_name, open_time, exit_time, exit_reason,
            open_mark, exit_mark, net_pnl, worst_mark
        Ordered by open_time.
        """
        return (
            self._load_positions()
            .select(
                [
                    "trade_name",
                    "open_time",
                    "exit_time",
                    "exit_reason",
                    "open_mark",
                    "exit_mark",
                    "net_pnl",
                    "worst_mark",
                ]
            )
            .sort("open_time")
        )

    def study_summary(self) -> pl.DataFrame:
        """
        Returns one row per backtest in the study, with key metrics as columns.
        Columns: backtest_id, combination_id, strategy_name, status,
                 net_profit, total_trades, percent_profitable, sharpe_ratio.

        Requires study_id to be set.
        """
        if self.study_id is None:
            raise ValueError("study_id is required for study_summary()")

        backtests = self.output_db._con.execute(
            """
            SELECT id, combination_id, strategy_name, status
            FROM backtest
            WHERE study_id = ?
            ORDER BY combination_id
            """,
            [self.study_id],
        ).pl()

        if backtests.is_empty():
            return pl.DataFrame()

        rows = []
        for row in backtests.iter_rows(named=True):
            bid = row["id"]
            pp = PostProcessor(self.output_db, backtest_id=bid)
            m = pp.metrics()
            rows.append({
                "backtest_id": bid,
                "combination_id": row["combination_id"],
                "strategy_name": row["strategy_name"],
                "status": row["status"],
                "net_profit": m["net_profit"],
                "total_trades": m["total_trades"],
                "percent_profitable": m["percent_profitable"],
                "sharpe_ratio": m["sharpe_ratio"],
                "profit_factor": m["profit_factor"] if math.isfinite(m["profit_factor"]) else None,
                "max_drawdown_pct": m["max_drawdown_pct"],
            })
        return pl.DataFrame(rows)

    def summarize(self, formatted: bool = False) -> str:
        """
        Returns a formatted multi-line string of all metrics for terminal display.
        """
        bid = self.backtest_id or "latest"
        m = self.metrics()

        def _fmt_ratio(v: float) -> str:
            return f"{v:>10.2f}" if math.isfinite(v) else "       inf"

        lines = [
            f"=== Backtest Results (id={bid}) ===",
            f"  Net Profit:           ${m['net_profit']:>10.2f}",
            f"  Total Trades:         {m['total_trades']:>10}",
            f"  Win Rate:             {m['percent_profitable'] * 100:>9.1f}%",
            f"  Profit Factor:        {_fmt_ratio(m['profit_factor'])}",
            f"  Avg Win:              ${m['avg_win']:>10.2f}",
            f"  Avg Loss:             ${m['avg_loss']:>10.2f}",
            f"  Median PnL:           ${m['median_pnl']:>10.2f}",
            f"  Max Drawdown:         ${m['max_drawdown']:>10.2f}",
            f"  Max Drawdown %:       {m['max_drawdown_pct'] * 100:>9.2f}%",
            f"  CAGR:                 {m['cagr'] * 100:>9.2f}%",
            f"  MAR Ratio:            {_fmt_ratio(m['mar'])}",
            f"  Sharpe Ratio:         {m['sharpe_ratio']:>10.2f}",
            f"  Sortino Ratio:        {_fmt_ratio(m['sortino_ratio'])}",
            f"  Calmar Ratio:         {_fmt_ratio(m['calmar_ratio'])}",
            f"  Premium Capture:      {m['premium_capture_rate'] * 100:>9.2f}%",
            f"  Avg MAE:              ${m['avg_mae']:>10.2f}",
            f"  Median MAE:           ${m['median_mae']:>10.2f}",
            f"  Worst MAE:            ${m['worst_mae']:>10.2f}",
        ]
        return "\n".join(lines)

    def heatmap(
        self,
        metric: str,
        x_param: str,
        y_param: str,
        fixed_params: dict[str, Any] | None = None,
    ) -> Any:
        """
        Plot a heatmap of the given metric across two swept parameters.
        Requires study_id and a StudyExpander.params_df to be provided.
        fixed_params filters the study to a specific slice when more than
        two parameters were swept.
        """
        raise NotImplementedError(
            "heatmap() requires study_id and params_df — not yet implemented"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_positions(self) -> pl.DataFrame:
        """Load positions for this backtest or study run."""
        if self.study_id is not None:
            return self.output_db._con.execute(
                """
                SELECT p.* FROM position p
                JOIN backtest b ON p.backtest_id = b.id
                WHERE b.study_id = ?
                """,
                [self.study_id],
            ).pl()
        bid = self._resolve_backtest_id()
        if bid is None:
            return pl.DataFrame()
        return self.output_db._con.execute(
            "SELECT * FROM position WHERE backtest_id = ?", [bid]
        ).pl()

    def _load_initial_equity(self) -> float:
        bid = self._resolve_backtest_id()
        if bid is None:
            return 100_000.0
        row = self.output_db._con.execute(
            "SELECT initial_equity FROM backtest WHERE id = ?", [bid]
        ).fetchone()
        return float(row[0]) if row else 100_000.0

    def _resolve_backtest_id(self) -> int | None:
        """Return backtest_id, defaulting to the most recent if not set."""
        if self.backtest_id is not None:
            return self.backtest_id
        row = self.output_db._con.execute(
            "SELECT id FROM backtest ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def _max_drawdown(self, equity_df: pl.DataFrame, initial_equity: float) -> tuple[float, float]:
        if equity_df.is_empty():
            return 0.0, 0.0
        equities = equity_df["equity"].to_list()
        peak = initial_equity
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = max_dd / peak if peak > 0 else 0.0
        return max_dd, max_dd_pct

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "net_profit": 0.0,
            "total_trades": 0,
            "percent_profitable": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "median_pnl": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "cagr": 0.0,
            "mar": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "premium_capture_rate": 0.0,
            "avg_mae": 0.0,
            "median_mae": 0.0,
            "worst_mae": 0.0,
        }
