import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import sqlite3


RISK_FREE_RATE = 0.01
TRADING_DAYS_PER_YEAR = 252

class PostprocTool:
    
    def __init__(self, result_db_path: str, session_id: int, starting_balance):
        self.session_id = session_id
        self.starting_balance = starting_balance
        conn = sqlite3.connect(result_db_path)
        query = f"""
            SELECT * FROM trade WHERE session_id = {session_id}
        """
        df = pd.read_sql_query(query, conn)
        self.raw_trades_df = df
        
        # Calculate per-trade PnL
        df["cash_effect"] = df["action"].str.contains("SELL_TO_OPEN|BUY_TO_OPEN").astype(int).replace({0: -1}) * df["mkt_price"]
        self.agg_trades_df = (
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
        self.agg_trades_df["time"] = pd.to_datetime(self.agg_trades_df["time"], errors="coerce")
        self.agg_trades_df = self.agg_trades_df.dropna(subset=["time"]).sort_values("time")
        self.agg_trades_df["date"] = self.agg_trades_df["time"].dt.date

        # Calculate equity over time
        self.agg_trades_df["equity"] = self.starting_balance + self.agg_trades_df["pnl"].cumsum()
        
        
    def equity_curve(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.agg_trades_df["time"], self.agg_trades_df["equity"], lw=2)
        ax.set_title("Strategy Equity Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Account Value")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    
    def pnl_histogram(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.agg_trades_df["pnl"], bins=50, edgecolor="black")
        plt.title("Distribution of Trade PnL")
        plt.xlabel("PnL")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.show()
        
    
    def pnl_scatterplot(self):
        # Map each unique date to an integer index (equal spacing)
        unique_dates = sorted(self.agg_trades_df["date"].unique())
        date_to_index = {d: i for i, d in enumerate(unique_dates)}
        self.agg_trades_df["date_index"] = self.agg_trades_df["date"].map(date_to_index)

        # --- Base scatter plot ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(self.agg_trades_df["date_index"], self.agg_trades_df["pnl"], alpha=0.6)
        ax.set_title("Trade PnL vs Time (Equal Spacing by Trade Day)")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL")
        ax.grid(alpha=0.3, axis="both")

        # --- Top tick labels: every-other day ---
        tick_positions = range(len(unique_dates))
        day_labels = [pd.to_datetime(d).strftime("%d") for d in unique_dates]

        # Keep every other day labeled
        visible_ticks = [i for i in tick_positions if i % 2 == 0]
        visible_labels = [day_labels[i] for i in visible_ticks]

        ax.set_xticks(visible_ticks)
        ax.set_xticklabels(visible_labels, rotation=0)

        # --- Add second (month) tier below ---
        date_series = pd.to_datetime(unique_dates)
        months = date_series.to_period("M")

        month_labels = []
        for month, idxs in pd.Series(months).groupby(months):
            start = idxs.index.min()
            end = idxs.index.max()
            center = (start + end) / 2
            month_labels.append((center, month.strftime("%b")))

        # Add month names below the day ticks
        y_min, y_max = ax.get_ylim()
        for center, label in month_labels:
            ax.text(center, y_min - (y_max - y_min) * 0.05,
                    label, ha="center", va="top", fontsize=10)

        # Keep tick lines visible (restore default tick length)
        ax.tick_params(axis="x", which="both", length=4)

        # Calculate the average PnL across all trades
        avg_pnl = self.agg_trades_df["pnl"].mean()

        # Plot the average line
        ax.axhline(
            y=avg_pnl,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Average PnL = {avg_pnl:.2f}"
        )

        # Add a legend so it's labeled on the chart
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()
        
        
    def strategy_metrics(self):
        # --- Basic stats ---
        net_profit = self.agg_trades_df["pnl"].sum()
        total_closed_trades = len(self.agg_trades_df)
        percent_profitable_trades = (self.agg_trades_df["pnl"] > 0).mean() * 100
        median_trade_pnl = self.agg_trades_df["pnl"].median()
        average_trade_pnl = self.agg_trades_df["pnl"].mean()

        # --- Profit factor ---
        gross_profit = self.agg_trades_df.loc[self.agg_trades_df["pnl"] > 0, "pnl"].sum()
        gross_loss = -self.agg_trades_df.loc[self.agg_trades_df["pnl"] < 0, "pnl"].sum()  # make positive
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

        # --- Drawdown calculation ---
        equity = self.agg_trades_df["equity"]
        running_max = equity.cummax()
        drawdowns = running_max - equity
        max_drawdown = drawdowns.max()

        # --- Sharpe ratio ---
        # Using per-trade returns relative to equity before trade
        # r_i = PnL / equity_before_trade
        equity_shifted = equity.shift(1).fillna(equity.iloc[0])
        returns = self.agg_trades_df["pnl"] / equity_shifted
        excess_returns = returns - 0.01 / 252  # convert annual RF to daily approx (assuming trades ~1/day)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # annualized

        # --- Sortino ratio ---
        neg_returns = excess_returns[excess_returns < 0]
        downside_std = neg_returns.std()
        sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252)

        # --- Calmar ratio ---
        # Annualized return / max drawdown
        # Approximate annualized return using cumulative net profit / starting balance
        starting_balance = self.agg_trades_df["equity"].iloc[0]
        cumulative_return = (equity.iloc[-1] - starting_balance) / starting_balance
        # Estimate years elapsed based on days between first and last trade
        days_elapsed = (self.agg_trades_df["time"].iloc[-1] - self.agg_trades_df["time"].iloc[0]).days
        years_elapsed = days_elapsed / 365.25 if days_elapsed > 0 else 1/252  # avoid divide by zero
        annualized_return = (1 + cumulative_return) ** (1 / years_elapsed) - 1
        calmar_ratio = annualized_return / (max_drawdown / starting_balance)

        # --- Print results ---
        print("======================================================")
        print(f"Net Profit: ${net_profit:.2f}")
        print(f"Total Closed Trades: {total_closed_trades}")
        print(f"Percent Profitable Trades: {percent_profitable_trades:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Median Trade PnL: ${median_trade_pnl:.2f}")
        print(f"Average Trade PnL: ${average_trade_pnl:.2f}")
        print(f"Maximum Drawdown: ${max_drawdown:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print("======================================================")


        # self.agg_trades_df["daily_return"] = self.agg_trades_df["pnl"] / self.agg_trades_df["equity"]
        # wins = self.agg_trades_df[self.agg_trades_df["pnl"] > 0]
        # losses = self.agg_trades_df[self.agg_trades_df["pnl"] <= 0]
        # total_pnl = self.agg_trades_df["pnl"].sum()
        # total_trades = self.agg_trades_df.shape[0]
        # pct_profit = 100 * (wins.shape[0] / self.agg_trades_df.shape[0])
        
        # profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum())
        # median_pnl = self.agg_trades_df["pnl"].median()
        # avg_pnl = self.agg_trades_df["pnl"].mean()
        # max_drawdown = self.agg_trades_df["account_pnl"].min()
        # avg_daily_return = self.agg_trades_df["daily_return"].mean()
        # std_all = self.agg_trades_df["daily_return"].std()        
        # std_loss = losses["daily_return"].std()
        # daily_rfr = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        # excess_return = avg_daily_return - daily_rfr
        
        # #avg_time_in = pnl_df["BarsInTrade"].mean() / 60 # convert to minutes
        
        # # NOTE: Multiplying by 252**.5 should effectively annualize these values since,
        # # by convention, sharpe and sortino ratio are expressed in annualized terms.
        # # see https://quant.stackexchange.com/questions/57640/question-regarding-sharpe-ratio-calculation
        # sharpe = (excess_return / std_all) * TRADING_DAYS_PER_YEAR**.5
        # sortino = (excess_return / std_loss * TRADING_DAYS_PER_YEAR**.5)
        # calmar = (avg_daily_return * TRADING_DAYS_PER_YEAR) / max_drawdown
        # lines = [
        #     f"======================================================",
        #     f"Net Profit:\t\t${total_pnl:.2f}",
        #     f"Total Closed Trades:\t{total_trades}",
        #     f"Percent Profitable:\t{pct_profit:.2f}%",
        #     f"Profit Factor:\t\t{profit_factor:.2f}",
        #     #f"Avg Time in Trade:\t{avg_time_in:.2f} minutes",
        #     f"Max Drawdown:\t\t-${abs(max_drawdown):.2f}",
        #     f"Median Trade PnL:\t${median_pnl:.2f}",
        #     f"Avg Trade PnL:\t\t${avg_pnl:.2f}",
        #     f"Sharpe Ratio:\t\t{sharpe:.2f}",
        #     f"Sortino Ratio:\t\t{sortino:.2f}",
        #     f"Calmar Ratio:\t\t{calmar:.2f}",
        #     f"======================================================"
        # ]
        # print("\n".join(lines))