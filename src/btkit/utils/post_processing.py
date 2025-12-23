import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from zoneinfo import ZoneInfo


RISK_FREE_RATE = 0.01
TRADING_DAYS_PER_YEAR = 252

class PostprocTool:
    def __init__(self, result_db_path: str, backtest_id: int | None = None):
        self.conn = duckdb.connect(result_db_path)
        #self.conn.execute("PRAGMA threads=auto;")

        # ------------------------------------------------------------
        # 1. Load backtest base table (authoritative config source)
        # ------------------------------------------------------------
        bt_filter = f"WHERE id = {backtest_id}" if backtest_id else ""
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

        # ------------------------------------------------------------
        # 2. Check if metrics table exists
        # ------------------------------------------------------------
        # metrics_exist = self.conn.execute("""
        #     SELECT COUNT(*) > 0
        #     FROM information_schema.tables
        #     WHERE table_name = 'backtest_metrics'
        # """).fetchone()[0]
        metrics_exist = True

        # ============================================================
        # FAST PATH: metrics already exist
        # ============================================================
        if metrics_exist:
            self.backtest_df = bt_df
            # m_filter = f"WHERE id = {backtest_id}" if backtest_id else ""
            # metrics_df = self.conn.execute(
            #     f"""
            #     SELECT
            #         id,strategy_name,strategy_version,starting_cash,start_time,end_time,run_error,design_id,call_delta,call_spread_widths,dte,put_delta,put_spread_widths,strategy_type,take_profit_pct,net_profit,total_closed_trades,percent_profitable,median_trade_pnl,average_trade_pnl,average_win,average_loss,profit_factor,max_drawdown,cagr,mar,sharpe_ratio,sortino_ratio,calmar_ratio
            #     FROM backtest_metrics
            #     {m_filter}
            #     """
            # ).fetchdf()

            # self.backtest_df = bt_df.merge(
            #     metrics_df,
            #     on="id",
            #     how="left"
            # )
            return

        # ============================================================
        # SLOW PATH: compute metrics
        # ============================================================

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
        # 5. Drawdowns
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
            ) - equity AS drawdown
        FROM equity_curve
        """)

        # ------------------------------------------------------------
        # 6. Backtest-level metrics (authoritative results source)
        # ------------------------------------------------------------
        metrics_df = self.conn.execute("""
        SELECT
            backtest_id AS id,

            SUM(pnl) AS net_profit,
            COUNT(*) AS total_closed_trades,
            AVG(pnl > 0) * 100 AS percent_profitable,
            MEDIAN(pnl) AS median_trade_pnl,
            AVG(pnl) AS average_trade_pnl,

            AVG(CASE WHEN pnl > 0 THEN pnl END) AS average_win,
            AVG(CASE WHEN pnl < 0 THEN pnl END) AS average_loss,

            SUM(CASE WHEN pnl > 0 THEN pnl END)
              / NULLIF(-SUM(CASE WHEN pnl < 0 THEN pnl END), 0)
              AS profit_factor,

            MAX(drawdown) AS max_drawdown,

            MIN(time) AS start_time,
            MAX(time) AS end_time,
            FIRST(equity) AS start_equity,
            LAST(equity) AS end_equity

        FROM drawdowns
        GROUP BY backtest_id
        """).fetchdf()

        # ------------------------------------------------------------
        # 7. Python-side metrics (derived only from metrics_df)
        # ------------------------------------------------------------
        metrics_df["start_time"] = pd.to_datetime(
            metrics_df["start_time"], utc=True
        ).dt.tz_convert(ZoneInfo("America/New_York"))

        metrics_df["end_time"] = pd.to_datetime(
            metrics_df["end_time"], utc=True
        ).dt.tz_convert(ZoneInfo("America/New_York"))

        years = (
            (metrics_df["end_time"] - metrics_df["start_time"])
            .dt.days / 365.25
        )

        metrics_df["cagr"] = np.where(
            years > 0,
            (metrics_df["end_equity"] / metrics_df["start_equity"])
            ** (1 / years) - 1,
            np.nan
        )

        metrics_df["mar"] = metrics_df["cagr"] / metrics_df["max_drawdown"].abs()

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

    # ------------------------------------------------------------
    # On-demand equity curve for visualization
    # ------------------------------------------------------------
    def get_equity_curve(self, backtest_id: int) -> pd.DataFrame:
        df = self.conn.execute("""
        SELECT time, equity
        FROM equity_curve
        WHERE backtest_id = ?
        ORDER BY time
        """, [backtest_id]).fetchdf()

        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(
            ZoneInfo("America/New_York")
        )
        return df

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
        
        
    # TODO: ability to add comparisons to other series
    def equity_curve(self, backtest_id: int, show: bool = True, fig: go.Figure = None, **kwargs) -> go.Figure:
        df = self.trade_summaries[backtest_id]
        
        line_plot = go.Line(x=df["time"], y=df["equity"])
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Session {backtest_id} Equity Curve" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Time" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Equity (USD)" }),
            )
            
        fig.add_trace(line_plot)
        
        if show:
            fig.show()
        
        return fig
    
    
    def pnl_histogram(self, backtest_id: int, show: bool = True, fig: go.Figure = None, **kwargs):
        df = self.trade_summaries[backtest_id]
        
        histogram = go.Histogram(x=df["pnl"])
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Session {backtest_id} PnL Distribution" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Profit and Loss (USD)" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Occurrences" }),
            )
            
        fig.add_trace(histogram)
        
        if show:
            fig.show()
        
        return fig
    
    
    def trade_scatterplot(self, backtest_id: int, show: bool = True, subplot: bool = False, **kwargs):
        df = self.trade_summaries[backtest_id]
        
        scatter_plot = go.Scatter(x=df["time"], y=df["pnl"], mode="markers", name=kwargs.get("name", "Trade PnL"))
        
        # Add to figure and show
        if not subplot:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Session {backtest_id} Trade PnL" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Time" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Profit and Loss (USD)" }),
            )
            fig.add_trace(scatter_plot)
            fig.show()
            return None

        return scatter_plot
        

    def heatmap(self, metric: str, selectors: dict[str, any], x_variable: str, y_variable: str, show: bool = False, fig: go.Figure = None, **kwargs):
        # Filter the data
        filtered_df = self.backtest_df.copy()
        for col, val in selectors.items():
            filtered_df = filtered_df[filtered_df[col] == val]
        
        # Pivot the dataframe and gather text and min/max z values
        pivot = filtered_df.pivot(index=x_variable, columns=y_variable, values=metric)
        z = pivot.values
        text = [[f"{v:.1f}" if pd.notna(v) else "" for v in row] for row in z]
        zmin, zmax = float(pivot.min().min()), float(pivot.max().max())
        
        custom = filtered_df.pivot(index=y_variable, columns=x_variable, values="id").values
        
        # Configure the heatmap
        heatmap = go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
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
                #xaxis=dict(type="category"),
                #yaxis=dict(type="category"),
                xaxis_title = kwargs.get("xaxis_title", {'text' : x_variable.replace("_", " ").title() }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : y_variable.replace("_", " ").title() }),
            )
            
        fig.add_trace(heatmap)
        
        if show:
            fig.show()

        return fig
            