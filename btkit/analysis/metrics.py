"""
PostProcessor — loads backtest results and computes standard metrics.

Supports both single-run (backtest_id) and matrix-run (matrix_id) analysis.
For MVP, single-run analysis and terminal output are the priority. Matrix-run
heatmap analysis is deferred.

MAE (Maximum Adverse Excursion) is derived from worst_mark and open_mark stored
in the position table — no additional data collection needed.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from btkit.db.output_db import OutputDatabase


class PostProcessor:
    def __init__(
        self,
        output_db: OutputDatabase,
        backtest_id: int | None = None,
        matrix_id: int | None = None,
    ) -> None:
        """
        Initialise with exactly one of backtest_id (single run) or matrix_id
        (all runs from a matrix expansion).
        """
        if (backtest_id is None) == (matrix_id is None):
            raise ValueError("Provide exactly one of backtest_id or matrix_id")
        self.output_db = output_db
        self.backtest_id = backtest_id
        self.matrix_id = matrix_id

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
        Only valid when initialised with backtest_id.
        """
        raise NotImplementedError

    def equity_curve(self) -> pl.DataFrame:
        """
        Returns a DataFrame of cumulative equity over time.
        Columns: ts_event, equity (initial_equity + cumulative net_pnl at each exit).
        """
        raise NotImplementedError

    def trade_pnl_series(self) -> pl.DataFrame:
        """
        Returns one row per trade with columns:
            open_time, exit_time, exit_reason, net_pnl, worst_mark
        Ordered by open_time.
        """
        raise NotImplementedError

    def summarize(self, formatted: bool = False) -> pl.DataFrame:
        """
        Returns a single-row DataFrame of all metrics() values. If formatted=True,
        numeric values are rounded and percentage metrics are expressed as percentages
        for terminal display.
        """
        raise NotImplementedError

    def heatmap(
        self,
        metric: str,
        x_param: str,
        y_param: str,
        fixed_params: dict[str, Any] | None = None,
    ) -> Any:
        """
        DEFERRED — requires matrix_id and StrategyMatrix.params_df.

        Plot a heatmap of the given metric across two swept parameters.
        fixed_params filters the matrix to a specific slice when more than
        two parameters were swept.
        """
        raise NotImplementedError("heatmap() is deferred — not available in MVP")
