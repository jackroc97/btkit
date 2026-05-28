# Dashboard

## Overview

The btkit dashboard is a Dash web application launched with `btkit serve`:

```
btkit serve --output-db <path> [--input-db <path>] [--backtest-id N] [--port 8050]
```

There are two top-level views:

- **Backtests / Studies index** (`/`) — lists all runs with headline metrics; click a row
  to open its full dashboard.
- **Backtest dashboard** (`/?backtest_id=N`) — five panels: equity curve, bootstrap equity
  fan, P&L histogram, strategy metrics, and trade list.
- **Study dashboard** (`/?study_id=N`) — multi-curve equity chart and combination results
  grid for parameter sweeps.

Individual trade charts are served at `/chart/<position_id>` using Lightweight Charts JS.

---

## Equity Curve — Compare Feature

The **Compare** button (top-right of the equity curve card) opens a panel for overlaying
a second series on the equity curve. Three modes are available:

### Buy & Hold

Fetches daily close prices for a ticker via `yfinance` and plots cumulative P&L relative
to the first price in the backtest date range.

Requires `yfinance`:
```
pip install yfinance
```

### Live Trades (CSV)

Overlays a cumulative P&L curve from a CSV file of live trade results. Useful for
comparing a backtest equity curve against actual trading performance.

#### CSV Format

Two columns are required; all other columns are ignored. A header row is mandatory.

```csv
date,pnl
2026-01-15,125.00
2026-01-22,-48.50
2026-01-29,200.00
```

| Column | Type | Description |
|---|---|---|
| `date` | string | Trade close date. Accepts `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`. |
| `pnl` | float | Net P&L for the trade in dollars (positive = profit, negative = loss). |

Column names are matched case-insensitively. The `date` column may also be named
`datetime`, `time`, or `timestamp`. The `pnl` column may also be named `net_pnl`,
`profit`, or `realized_pnl`.

Each row represents one closed trade (or one day's aggregate P&L). The dashboard
accumulates rows in date order and plots the running total. Rows outside the
backtest's date range are excluded automatically.

#### Preparing the CSV

The CSV is broker-agnostic — prepare it in a spreadsheet, export from your broker's
reporting tool, or generate it programmatically. The only requirement is that P&L
values are net of commissions and represent realized results on closed positions.

Example from a spreadsheet export:
```csv
date,pnl
2026-01-15,312.50
2026-01-17,-125.00
2026-01-22,437.50
```

### Backtest

Overlays the equity curve of another backtest run from the same output database.
Select the run from the dropdown; the curve is fetched directly from the database.

---

## Trade Chart Pages

Each row in the trade list has a chart icon (if `--input-db` is provided). Clicking it
opens `/chart/<position_id>` — a full-screen Lightweight Charts page showing:

- Underlying OHLCV candlestick chart for the trade day(s)
- Entry and exit markers
- Breakeven price lines
- Individual strike reference lines
- Running P&L pane (synced to the candlestick time axis)
- Hypothetical post-exit P&L (grey dashed line)
- TP/SL dollar levels on the P&L pane

Trade charts require `--input-db` to be passed to `btkit serve`. Without it, a
watermark is shown and the underlying data panel is empty.
