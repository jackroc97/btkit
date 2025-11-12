import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3

from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo


RISK_FREE_RATE = 0.01
TRADING_DAYS_PER_YEAR = 252

class PostprocTool:
    def __init__(self, result_db_path: str, session_id: int = None):
        # Select all sessions from the database, or filtered by session_id if provided
        self.conn = sqlite3.connect(result_db_path)
        session_filter = f"WHERE session_id = {session_id}" if session_id else ""
        session_df = pd.read_sql_query(f"""
            SELECT * FROM session {session_filter}
        """, self.conn)
        
        # Unpack strategy parameters from json and merge into the session dataframe
        session_df["strategy_params"] = session_df["strategy_params"].apply(json.loads)
        json_expanded = pd.json_normalize(session_df["strategy_params"])
        self.session_df = pd.concat([session_df.drop(columns=["strategy_params"]), json_expanded], axis=1)

        self.trade_summaries: dict[int, pd.DataFrame] = dict()

        # Calculate results for all sessions
        results_df = pd.DataFrame()
        for _, row in self.session_df.iterrows():
            query = f"""
                SELECT * FROM trade WHERE session_id = {row['id']}
            """
            df = pd.read_sql_query(query, self.conn)
            
            # Calculate per-trade PnL
            df["cash_effect"] = df["action"].str.contains("SELL_TO_OPEN|BUY_TO_OPEN").astype(int).replace({0: -1}) * df["mkt_price"]
            df = (
                df.groupby("position_uuid", as_index=False)
                  .agg({
                        "time": "max",           # use latest trade time for plotting
                        "cash_effect": "sum",
                        "symbol": "first",
                        "expiration": "first",
                        "right": "first"
                  })
                  .rename(columns={"cash_effect": "pnl"})
            )
            
            # Compute time
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df["time"] = df["time"].dt.tz_convert(ZoneInfo("America/New_York"))
            df = df.dropna(subset=["time"]).sort_values("time")
            df["date"] = df["time"].dt.date
    
            # Calculate equity over time
            df["equity"] = row['starting_balance'] + df["pnl"].cumsum()
            
            self.trade_summaries[row['id']] = df
            
            # Basic stats
            net_profit = df["pnl"].sum()
            total_closed_trades = len(df)
            percent_profitable_trades = (df["pnl"] > 0).mean() * 100
            median_trade_pnl = df["pnl"].median()
            average_trade_pnl = df["pnl"].mean()
            
            average_pnl_win = df[df["pnl"] > 0]["pnl"].mean()
            average_pnl_loss = df[df["pnl"] < 0]["pnl"].mean()
            
            # Profit factor
            gross_profit = df.loc[df["pnl"] > 0, "pnl"].sum()
            gross_loss = -df.loc[df["pnl"] < 0, "pnl"].sum()  # make positive
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

            # Drawdown calculation
            equity = df["equity"]
            running_max = equity.cummax()
            drawdowns = running_max - equity
            max_drawdown = drawdowns.max()
            
            # CAGR
            start_equity = row['starting_balance']
            end_equity = equity.iloc[-1]
            total_days = (df["time"].iloc[-1] - df["time"].iloc[0]).days
            years = total_days / 365.25
            cagr = (end_equity / start_equity) ** (1 / years) - 1 if years > 0 else np.nan
            
            # MAR
            mar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
            
            # Sharpe ratio
            # Using per-trade returns relative to equity before trade
            # r_i = PnL / equity_before_trade
            equity_shifted = equity.shift(1).fillna(equity.iloc[0])
            returns = df["pnl"] / equity_shifted
            excess_returns = returns - 0.01 / 252  # convert annual RF to daily approx (assuming trades ~1/day)
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # annualized

            # Sortino ratio
            neg_returns = excess_returns[excess_returns < 0]
            downside_std = neg_returns.std()
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252)

            # Calmar ratio
            # Annualized return / max drawdown
            # Approximate annualized return using cumulative net profit / starting balance
            starting_balance = df["equity"].iloc[0]
            cumulative_return = (equity.iloc[-1] - starting_balance) / starting_balance
            annualized_return = (1 + cumulative_return) ** (1 / years) - 1
            calmar_ratio = annualized_return / (max_drawdown / starting_balance)
            
            # Combine into a dataframe
            results_df = pd.concat([results_df, pd.DataFrame({
                "id": [row["id"]],
                "net_profit": [net_profit],
                "total_closed_trades": [total_closed_trades],
                "percent_profitable": [percent_profitable_trades],
                "median_trade_pnl": [median_trade_pnl],
                "average_trade_pnl": [average_trade_pnl],
                "average_win": [average_pnl_win],
                "average_loss": [average_pnl_loss],
                "profit_factor": [profit_factor],
                "max_drawdown": [max_drawdown],
                "cagr": [cagr],
                "mar": [mar],
                "sharpe_ratio": [sharpe_ratio],
                "sortino_ratio": [sortino_ratio],
                "calmar_ratio": [calmar_ratio]
            })])
            
        # Merge stats into session dataframe
        self.session_df = pd.merge(self.session_df, results_df, on="id")
    
    
    def summarize(self, session_id: int):
        session = self.session_df[self.session_df["id"] == session_id].iloc[0]
        print("======================================================")
        print(f"Summary for Session {session['id']}")
        print("======================================================")
        print(f"Net Profit: ${session['net_profit']:.2f}")
        print(f"Total Closed Trades: {session['total_closed_trades']}")
        print(f"Percent Profitable Trades: {session['percent_profitable']:.2f}%")
        print(f"Profit Factor: {session['profit_factor']:.2f}")
        print(f"Median Trade PnL: ${session['median_trade_pnl']:.2f}")
        print(f"Average Trade PnL: ${session['average_trade_pnl']:.2f}")
        print(f"Average Win: ${session['average_win']:.2f}")
        print(f"Average Loss: ${session['average_loss']:.2f}")
        print(f"Maximum Drawdown: ${session['max_drawdown']:.2f}")
        print(f"CAGR: {(session['cagr'] * 100):.2f}%")
        print(f"MAR: {session['mar']:.2f}")
        print(f"Sharpe Ratio: {session['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {session['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {session['calmar_ratio']:.2f}")
        print("======================================================")
        
        
    # TODO: ability to add comparisons to other series
    def equity_curve(self, session_id: int, show: bool = True, fig: go.Figure = None, **kwargs) -> go.Figure:
        df = self.trade_summaries[session_id]
        
        line_plot = go.Line(x=df["time"], y=df["equity"])
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Session {session_id} Equity Curve" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Time" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Equity (USD)" }),
            )
            
        fig.add_trace(line_plot)
        
        if show:
            fig.show()
        
        return fig
    
    
    def pnl_histogram(self, session_id: int, show: bool = True, fig: go.Figure = None, **kwargs):
        df = self.trade_summaries[session_id]
        
        histogram = go.Histogram(x=df["pnl"])
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Session {session_id} PnL Distribution" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Profit and Loss (USD)" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Occurrences" }),
            )
            
        fig.add_trace(histogram)
        
        if show:
            fig.show()
        
        return fig
    
    
    def trade_scatterplot(self, session_id: int, show: bool = True, fig: go.Figure = None, **kwargs):
        df = self.trade_summaries[session_id]
        
        scatter_plot = go.Scatter(x=df["time"], y=df["pnl"], mode="markers")
        
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': f"Session {session_id} Trade PnL" }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : "Time" }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : "Profit and Loss (USD)" }),
            )
            
        fig.add_trace(scatter_plot)
        
        if show:
            fig.show()
        
        return fig
        

    def heatmap(self, metric: str, selectors: dict[str, any], x_variable: str, y_variable: str, show: bool = True, fig: go.Figure = None, **kwargs):
        # Filter the data
        filtered_df = self.session_df.copy()
        for col, val in selectors.items():
            filtered_df = filtered_df[filtered_df[col] == val]
        
        # Pivot the dataframe and gather text and min/max z values
        pivot = filtered_df.pivot(index=x_variable, columns=y_variable, values=metric)
        z = pivot.values
        text = [[f"{v:.1f}" if pd.notna(v) else "" for v in row] for row in z]
        zmin, zmax = float(pivot.min().min()), float(pivot.max().max())
        
        # Configure the heatmap
        heatmap = go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
            colorscale=kwargs.get("colorscale", [[0.0, "red"], [0.5, "white"], [1.0, "blue"]]),
            zmin=zmin,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": "black"},
        )

        # Add to figure and show
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title = kwargs.get("title", { 'text': metric.replace("_", " ").title() }),
                xaxis_title = kwargs.get("xaxis_title", {'text' : x_variable.replace("_", " ").title() }),
                yaxis_title = kwargs.get("yaxis_title", {'text' : y_variable.replace("_", " ").title() }),
            )
            
        fig.add_trace(heatmap)
        
        if show:
            fig.show()
            
        return fig
            