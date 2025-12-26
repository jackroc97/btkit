import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from zoneinfo import ZoneInfo


RISK_FREE_RATE = 0.01
TRADING_DAYS_PER_YEAR = 252

class PostProcTool:
    def __init__(self, result_db_path: str, backtest_id: int | None = None):
        self.conn = duckdb.connect(result_db_path)

        bt_filter = f"WHERE id = {backtest_id}" if backtest_id else ""

        # ============================================================
        # FAST PATH: metrics already exist
        # ============================================================
        metrics_exist = len(self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_metrics';"
        ).fetchall()) > 0
        if metrics_exist:
            bt_df = self.conn.execute(
                f"""
                SELECT
                    backtest.id,
                    backtest.starting_cash,
                    backtest.strategy_params,
                    backtest_metrics.net_profit,
                    backtest_metrics.total_closed_trades,
                    backtest_metrics.percent_profitable,
                    backtest_metrics.median_trade_pnl,
                    backtest_metrics.average_trade_pnl,
                    backtest_metrics.average_win,
                    backtest_metrics.average_loss,
                    backtest_metrics.profit_factor,
                    backtest_metrics.max_drawdown,
                    backtest_metrics.max_drawdown_decimal,
                    backtest_metrics.cagr,
                    backtest_metrics.mar,
                    backtest_metrics.sharpe_ratio,
                    backtest_metrics.sortino_ratio,
                    backtest_metrics.calmar_ratio
                FROM backtest
                {bt_filter}
                JOIN backtest_metrics ON backtest.id = backtest_metrics.id
                """
            ).fetchdf()

            # Expand strategy_params JSON → columns
            params_df = pd.json_normalize(bt_df["strategy_params"])
            bt_df = pd.concat(
                [bt_df.drop(columns=["strategy_params"]), params_df],
                axis=1
            )
            self.backtest_df = bt_df
            return

        # ============================================================
        # SLOW PATH: compute metrics
        # ============================================================
        bt_df = self.conn.execute(
            f"""
            SELECT
                id,
                starting_cash,
                strategy_params
            FROM backtest
            {bt_filter}
            """
        ).fetchdf()
        params_df = pd.json_normalize(bt_df["strategy_params"])
        bt_df = pd.concat(
            [bt_df.drop(columns=["strategy_params"]), params_df],
            axis=1
        )

        # ------------------------------------------------------------
        # 3. Pre-aggregate trades
        # ------------------------------------------------------------
        self.conn.execute("""
        CREATE OR REPLACE TEMP VIEW trade_pnl AS
        SELECT
            backtest_id,
            position_uuid,
            MAX(time) AS time,
            SUM(
                CASE
                    WHEN action LIKE '%SELL_TO_OPEN%'
                      OR action LIKE '%BUY_TO_OPEN%'
                    THEN mkt_price
                    ELSE -mkt_price
                END
            ) AS pnl
        FROM trade
        GROUP BY backtest_id, position_uuid
        """)

        # ------------------------------------------------------------
        # 4. Equity curve
        # ------------------------------------------------------------
        self.conn.execute("""
        CREATE OR REPLACE TEMP VIEW equity_curve AS
        SELECT
            t.backtest_id,
            t.time,
            t.pnl,
            b.starting_cash
              + SUM(t.pnl) OVER (
                    PARTITION BY t.backtest_id
                    ORDER BY t.time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS equity
        FROM trade_pnl t
        JOIN backtest b ON b.id = t.backtest_id
        """)

        # ------------------------------------------------------------
        # 5. Drawdowns (include decimal max drawdown)
        # ------------------------------------------------------------
        self.conn.execute("""
        CREATE OR REPLACE TEMP VIEW drawdowns AS
        SELECT
            backtest_id,
            time,
            pnl,
            equity,
            MAX(equity) OVER (
                PARTITION BY backtest_id
                ORDER BY time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) - equity AS drawdown,
            (MAX(equity) OVER (
                PARTITION BY backtest_id
                ORDER BY time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) - equity)
            / MAX(equity) OVER (PARTITION BY backtest_id) AS drawdown_decimal
        FROM equity_curve
        """)

        # ------------------------------------------------------------
        # 6. Backtest-level metrics (include decimal max drawdown)
        # ------------------------------------------------------------
        metrics_df = self.conn.execute("""
        SELECT
            backtest_id AS id,

            SUM(pnl) AS net_profit,
            COUNT(*) AS total_closed_trades,
            AVG((pnl > 0)::DOUBLE) * 100 AS percent_profitable,
            MEDIAN(pnl) AS median_trade_pnl,
            AVG(pnl) AS average_trade_pnl,

            AVG(CASE WHEN pnl > 0 THEN pnl END) AS average_win,
            AVG(CASE WHEN pnl < 0 THEN pnl END) AS average_loss,

            SUM(CASE WHEN pnl > 0 THEN pnl END)
              / NULLIF(-SUM(CASE WHEN pnl < 0 THEN pnl END), 0)
              AS profit_factor,

            MAX(drawdown) AS max_drawdown,
            MAX(drawdown_decimal) AS max_drawdown_decimal,

            MIN(time) AS start_time,
            MAX(time) AS end_time,
            FIRST(equity) AS start_equity,
            LAST(equity) AS end_equity

        FROM drawdowns
        GROUP BY backtest_id
        """).fetchdf()

        # ------------------------------------------------------------
        # 7. Python-side metrics
        # ------------------------------------------------------------
        metrics_df["start_time"] = pd.to_datetime(metrics_df["start_time"], utc=True).dt.tz_convert(ZoneInfo("America/New_York"))
        metrics_df["end_time"] = pd.to_datetime(metrics_df["end_time"], utc=True).dt.tz_convert(ZoneInfo("America/New_York"))

        years = ((metrics_df["end_time"] - metrics_df["start_time"]).dt.days / 365.25)
        metrics_df["cagr"] = np.where(
            years > 0,
            (metrics_df["end_equity"] / metrics_df["start_equity"]) ** (1 / years) - 1,
            np.nan
        )

        # MAR
        metrics_df["mar"] = metrics_df["cagr"] / metrics_df["max_drawdown_decimal"]

        # ------------------------------------------------------------
        # 7a. Compute Sharpe, Sortino using per-trade returns
        # ------------------------------------------------------------
        trade_df = self.conn.execute(f"""
            SELECT
                backtest_id,
                pnl,
                equity
            FROM equity_curve
            {f'WHERE backtest_id = {backtest_id}' if backtest_id else ''}
        """).fetchdf()

        # Returns relative to equity before trade
        trade_df["equity_before"] = trade_df.groupby("backtest_id")["equity"].shift(1)
        trade_df["equity_before"] = trade_df["equity_before"].fillna(trade_df.groupby("backtest_id")["equity"].transform("first"))
        trade_df["returns"] = trade_df["pnl"] / trade_df["equity_before"]

        rf_daily = 0.01 / 252  # approximate daily risk-free rate
        trade_df["excess_returns"] = trade_df["returns"] - rf_daily

        # Compute per-backtest Sharpe and Sortino ratios
        ratios_df = trade_df.groupby("backtest_id").apply(lambda g: pd.Series({
            "sharpe_ratio": (g["excess_returns"].mean() / g["excess_returns"].std()) * np.sqrt(252),
            "sortino_ratio": (g["excess_returns"].mean() / g["excess_returns"].loc[g["excess_returns"] < 0].std()) * np.sqrt(252)
        })).reset_index()

        # Merge ratios into metrics_df
        metrics_df = metrics_df.merge(ratios_df, left_on="id", right_on="backtest_id", how="left")
        metrics_df.drop(columns=["backtest_id"], inplace=True)

        # Calmar ratio
        metrics_df["calmar_ratio"] = metrics_df["cagr"] / metrics_df["max_drawdown_decimal"]

        # ------------------------------------------------------------
        # 8. Persist metrics (results-only table)
        # ------------------------------------------------------------
        self.conn.execute("""
        CREATE TABLE backtest_metrics AS
        SELECT * FROM metrics_df
        LIMIT 0
        """)
        self.conn.register("metrics_df", metrics_df)
        self.conn.execute("""
        INSERT INTO backtest_metrics
        SELECT * FROM metrics_df
        """)
        self.conn.execute("""
        CREATE UNIQUE INDEX idx_backtest_metrics_id
        ON backtest_metrics(id)
        """)

        # ------------------------------------------------------------
        # 9. Merge config + metrics (no overlap by design)
        # ------------------------------------------------------------
        self.backtest_df = bt_df.merge(
            metrics_df,
            on="id",
            how="left"
        )


    def summarize(self, backtest_id: int):
        backtest = self.backtest_df[self.backtest_df["id"] == backtest_id].iloc[0]
        print("======================================================")
        print(f"Summary for Session {backtest['id']}")
        print("======================================================")
        print(f"Net Profit: ${backtest['net_profit']:.2f}")
        print(f"Total Closed Trades: {backtest['total_closed_trades']}")
        print(f"Percent Profitable Trades: {backtest['percent_profitable']:.2f}%")
        print(f"Profit Factor: {backtest['profit_factor']:.2f}")
        print(f"Median Trade PnL: ${backtest['median_trade_pnl']:.2f}")
        print(f"Average Trade PnL: ${backtest['average_trade_pnl']:.2f}")
        print(f"Average Win: ${backtest['average_win']:.2f}")
        print(f"Average Loss: ${backtest['average_loss']:.2f}")
        print(f"Maximum Drawdown: ${backtest['max_drawdown']:.2f}")
        print(f"CAGR: {(backtest['cagr'] * 100):.2f}%")
        print(f"MAR: {backtest['mar']:.2f}")
        print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {backtest['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {backtest['calmar_ratio']:.2f}")
        print("======================================================")
        

    def get_equity_over_time(self, backtest_ids: list[int]) -> pd.DataFrame:
        """
        Retrieve per-trade equity curves for a list of backtest IDs.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            An active DuckDB connection.
        backtest_ids : list[int]
            List of backtest IDs to generate equity curves for.

        Returns
        -------
        pd.DataFrame
            DataFrame containing backtest_id, time, position_uuid, pnl, symbol,
            expiration, right, and cumulative equity.
        """
        if not backtest_ids:
            return pd.DataFrame()  # Return empty if no backtests provided

        # Format the list for SQL IN clause
        ids_str = ",".join(str(i) for i in backtest_ids)

        query = f"""
        WITH selected_backtests AS (
            SELECT id, starting_cash
            FROM backtest
            WHERE id IN ({ids_str})
        ),
        trade_pnl AS (
            SELECT
                t.backtest_id,
                t.position_uuid,
                MAX(t.time) AS time,
                SUM(
                    CASE 
                        WHEN t.action LIKE '%SELL_TO_OPEN%' OR t.action LIKE '%BUY_TO_OPEN%'
                        THEN t.mkt_price
                        ELSE -t.mkt_price
                    END
                ) AS pnl,
                ANY_VALUE(t.symbol) AS symbol,
                ANY_VALUE(t.expiration) AS expiration,
                ANY_VALUE(t.right) AS right
            FROM trade t
            WHERE t.backtest_id IN (SELECT id FROM selected_backtests)
            GROUP BY t.backtest_id, t.position_uuid
        )
        SELECT
            t.backtest_id,
            t.time,
            t.position_uuid,
            t.pnl,
            t.symbol,
            t.expiration,
            t.right,
            s.starting_cash + SUM(t.pnl) OVER (
                PARTITION BY t.backtest_id
                ORDER BY t.time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS equity
        FROM trade_pnl t
        JOIN selected_backtests s ON t.backtest_id = s.id
        ORDER BY t.backtest_id, t.time;
        """

        return self.conn.execute(query).fetchdf()

        
    # TODO: ability to add comparisons to other series
    def equity_curve(self, backtest_ids: list[int], show: bool = True, fig: go.Figure = None, **kwargs) -> go.Figure:
        df = self.get_equity_over_time(backtest_ids)
        
        df = df.groupby("backtest_id")
        line_plots = []
        for backtest_id, group in df:
            line_plots.append(go.Scatter(x=group["time"], y=group["equity"], name=kwargs.get("name", f"Backtest {backtest_id}")))
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Backtests {','.join(str(id) for id in backtest_ids)} Equity Curve" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Time" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Equity (USD)" }),
            )
            
        for line_plot in line_plots:
            fig.add_trace(line_plot)
        
        if show:
            fig.show()
        
        return fig


    def pnl_histogram(self, backtest_ids: list[int], show: bool = True, fig: go.Figure = None, **kwargs) -> go.Figure:
        df = self.get_equity_over_time(backtest_ids)
        
        df = df.groupby("backtest_id")
        histograms = []
        for backtest_id, group in df:
            histograms.append(go.Histogram(x=group["pnl"], name=kwargs.get("name", f"Backtest {backtest_id}")))
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Backtests {','.join(str(id) for id in backtest_ids)} PnL Distribution" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Profit and Loss (USD)" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Occurrences" }),
            )
            
        for histogram in histograms:
            fig.add_trace(histogram)
        
        if show:
            fig.show()
        
        return fig


    def trade_scatterplot(self, backtest_ids: list[int], show: bool = True, fig: go.Figure = None, **kwargs) -> go.Figure:
        df = self.get_equity_over_time(backtest_ids)
        
        df = df.groupby("backtest_id")
        scatter_plots = []
        for backtest_id, group in df:
            scatter_plots.append(go.Scatter(x=group["time"], y=group["pnl"], mode="markers", name=kwargs.get("name", f"Backtest {backtest_id}")))
        
        # Add to figure and show
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Backtests {','.join(str(id) for id in backtest_ids)} Trade PnL" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Time" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Profit and Loss (USD)" }),
            )
            
        for scatter_plot in scatter_plots:
            fig.add_trace(scatter_plot)

        if show:
            fig.show()

        return fig
        

    def heatmap(self, metric: str, selectors: dict[str, any], x_variable: str, y_variable: str, show: bool = False, fig: go.Figure = None, **kwargs):
        # Filter the data
        filtered_df = self.backtest_df.copy()
        for col, val in selectors.items():
            filtered_df = filtered_df[filtered_df[col] == val]
        
        # Pivot the dataframe and gather text and min/max z values
        pivot = filtered_df.pivot(index=y_variable, columns=x_variable, values=metric)
        z = pivot.values
        text = [[f"{v:.1f}" if pd.notna(v) else "" for v in row] for row in z]
        zmin, zmax = float(pivot.min().min()), float(pivot.max().max())

        # Customdata aligned with z
        custom = filtered_df.pivot(index=y_variable, columns=x_variable, values="id").values
        custom = custom.astype(str)

        # Heatmap trace
        heatmap = go.Heatmap(
            z=z,
            x=pivot.columns,  # x_variable
            y=pivot.index,    # y_variable
            showscale=False,
            colorscale=kwargs.get("colorscale", [[0.0, "red"], [0.5, "white"], [1.0, "blue"]]),
            zmin=zmin,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": "black"},
            name=kwargs.get("name", metric.replace("_", " ").title()),
            customdata=custom,
            hovertemplate=
                "x: %{x}<br>" +
                "y: %{y}<br>" +
                "z: %{z}<br>" +
                "backtest_id: %{customdata}" +
                "<extra></extra>"
        )

        # Add to figure and show
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': metric.replace("_", " ").title() }),
                xaxis=dict(type="category"),
                yaxis=dict(type="category"),
                xaxis_title = kwargs.get("xaxis_title", {'text' : x_variable.replace("_", " ").title() }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : y_variable.replace("_", " ").title() }),
            )
            
        fig.add_trace(heatmap)
        
        if show:
            fig.show()

        return fig
            