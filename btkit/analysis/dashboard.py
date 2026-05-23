"""
Dash dashboard for btkit backtest results.

Five components:
  1. Equity curve — cumulative portfolio value over time
  2. Bootstrap equity curve fan — 1 000 resampled paths showing outcome distribution
  3. P&L histogram — all trades or wins/losses split
  4. Strategy metrics table
  5. Trade list with per-trade chart links (Lightweight Charts, opens in new tab)

Launch via:
    btkit serve --output-db <path> [--input-db <path>] [--backtest-id N] [--port 8050]

Trade chart pages are served at GET /chart/<position_id> and use the Lightweight
Charts JS library (the same library that lightweight-charts-python wraps) embedded
directly in a Flask route on the Dash server — no background process needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import dash_ag_grid as dag
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, dcc, html
from flask import Response

from btkit.analysis.metrics import PostProcessor
from btkit.db.output_db import OutputDatabase

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BLUE   = "#2563EB"
_GREEN  = "#16A34A"
_RED    = "#DC2626"
_AMBER  = "#D97706"
_GRAY   = "#6B7280"
_BG     = "#F3F4F6"
_CARD   = "#FFFFFF"
_BORDER = "#E5E7EB"
_TEXT   = "#111827"
_MUTED  = "#6B7280"
_HEADER = "#111827"

_ASSETS = str(Path(__file__).parent / "assets")

# ---------------------------------------------------------------------------
# Lightweight Charts HTML template (served per trade at /chart/<id>)
# ---------------------------------------------------------------------------

_CHART_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>btkit — {trade_name}</title>
  <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    *  {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Inter, system-ui, sans-serif; background: #fff; overflow: hidden; }}
    #hdr {{
      background: #111827; color: #fff; padding: 0 16px;
      font-size: 13px; display: flex; align-items: center; gap: 14px;
      height: 42px; flex-shrink: 0;
    }}
    #hdr b  {{ font-size: 15px; letter-spacing: .04em; }}
    #hdr .sep   {{ color: #4B5563; }}
    #hdr .muted {{ color: #9CA3AF; font-size: 12px; }}
    #hdr .pnl   {{ margin-left: auto; font-weight: 700; font-size: 14px; }}
    #hdr .reason {{ color: #9CA3AF; font-size: 12px; }}
    #chart {{ position: fixed; top: 42px; left: 0; right: 0; bottom: 0; }}
  </style>
</head>
<body>
<div id="hdr">
  <b>btkit</b>
  <span class="sep">·</span>
  <span>{trade_name}</span>
  <span class="muted">{open_str} → {exit_str}</span>
  <span class="reason">{exit_reason}</span>
  <span class="pnl" style="color:{pnl_color}">{pnl_str}</span>
</div>
<div id="chart"></div>
<script>
const data    = {candle_json};
const markers = {markers_json};

const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  layout: {{ background: {{ color: '#fff' }}, textColor: '#374151' }},
  grid:   {{ vertLines: {{ color: '#F3F4F6' }}, horzLines: {{ color: '#F3F4F6' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: '#E5E7EB' }},
  timeScale: {{ borderColor: '#E5E7EB', timeVisible: true, secondsVisible: false }},
  width:  document.getElementById('chart').clientWidth,
  height: document.getElementById('chart').clientHeight,
}});

const cs = chart.addCandlestickSeries({{
  upColor: '#16A34A', downColor: '#DC2626',
  borderUpColor: '#16A34A', borderDownColor: '#DC2626',
  wickUpColor: '#16A34A', wickDownColor: '#DC2626',
}});

if (data.length > 0) {{
  cs.setData(data);
  cs.setMarkers(markers);
  chart.timeScale().fitContent();
}} else {{
  chart.applyOptions({{
    watermark: {{
      visible: true, fontSize: 18, horzAlign: 'center', vertAlign: 'center',
      color: '#9CA3AF', text: 'No underlying bar data (pass --input-db to btkit serve)',
    }},
  }});
}}

window.addEventListener('resize', () => {{
  chart.applyOptions({{
    width:  document.getElementById('chart').clientWidth,
    height: document.getElementById('chart').clientHeight,
  }});
}});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _card(children, style: dict | None = None) -> html.Div:
    base: dict = {
        "backgroundColor": _CARD,
        "border": f"1px solid {_BORDER}",
        "borderRadius": "8px",
        "padding": "18px 20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.06)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def _section_label(text: str) -> html.P:
    return html.P(text, style={
        "fontSize": "11px", "fontWeight": "600",
        "letterSpacing": "0.08em", "textTransform": "uppercase",
        "color": _MUTED, "margin": "0 0 12px 0",
    })


# ---------------------------------------------------------------------------
# 1. Equity curve
# ---------------------------------------------------------------------------

def _build_equity_chart(equity_df: pl.DataFrame, initial_equity: float) -> go.Figure:
    if equity_df.is_empty():
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No trade data", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=14, color=_MUTED))],
            paper_bgcolor="white", plot_bgcolor="white",
        )
        return fig

    times    = equity_df["exit_time"].to_list()
    equities = equity_df["equity"].to_list()
    times    = [times[0]] + times
    equities = [initial_equity] + equities

    final    = equities[-1]
    line_clr = _GREEN if final >= initial_equity else _RED
    r, g, b  = int(line_clr[1:3], 16), int(line_clr[3:5], 16), int(line_clr[5:7], 16)

    eq_min  = min(equities + [initial_equity])
    eq_max  = max(equities + [initial_equity])
    padding = max((eq_max - eq_min) * 0.25, initial_equity * 0.005)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=[initial_equity] * len(times),
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=equities,
        mode="lines",
        line=dict(color=line_clr, width=2),
        fill="tonexty",
        fillcolor=f"rgba({r},{g},{b},0.12)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))
    fig.add_hline(
        y=initial_equity, line_dash="dot", line_color=_GRAY, line_width=1,
        annotation_text=f"Start  ${initial_equity:,.0f}",
        annotation_font=dict(size=11, color=_GRAY),
        annotation_position="bottom right",
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                   tickformat="%b %d", tickfont=dict(size=11), title=None, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                   tickformat="$,.0f", tickfont=dict(size=11), title=None, zeroline=False,
                   range=[eq_min - padding, eq_max + padding]),
        showlegend=False, hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Bootstrap equity curve fan
# ---------------------------------------------------------------------------

def _build_bootstrap_fig(
    net_pnls: list[float],
    initial_equity: float,
    n_iter: int = 1_000,
) -> go.Figure:
    n = len(net_pnls)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return fig

    arr = np.array(net_pnls, dtype=float)
    x   = list(range(n + 1))

    rng     = np.random.default_rng(42)
    samples = rng.choice(arr, size=(n_iter, n), replace=True)
    cum     = np.hstack([np.zeros((n_iter, 1)), np.cumsum(samples, axis=1)]) + initial_equity

    p5,  p10 = np.percentile(cum, 5,  axis=0), np.percentile(cum, 10, axis=0)
    p25, p50 = np.percentile(cum, 25, axis=0), np.percentile(cum, 50, axis=0)
    p75, p90 = np.percentile(cum, 75, axis=0), np.percentile(cum, 90, axis=0)
    p95      = np.percentile(cum, 95, axis=0)

    actual = np.concatenate([[0], np.cumsum(arr)]) + initial_equity
    line_clr = _GREEN if actual[-1] >= initial_equity else _RED

    fig = go.Figure()

    def _band(lo, hi, fill_alpha, name):
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=list(hi) + list(lo[::-1]),
            fill="toself", fillcolor=f"rgba(37,99,235,{fill_alpha})",
            line=dict(width=0), name=name, showlegend=True, hoverinfo="none",
        ))

    _band(p5,  p95, 0.06, "5th–95th %ile")
    _band(p10, p90, 0.08, "10th–90th %ile")
    _band(p25, p75, 0.12, "25th–75th %ile")

    fig.add_trace(go.Scatter(
        x=x, y=list(p50),
        mode="lines", line=dict(color=_BLUE, width=1.5, dash="dash"),
        name="Median", showlegend=True, hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=list(actual),
        mode="lines", line=dict(color=line_clr, width=2.5),
        name="Actual", showlegend=True,
        hovertemplate="Trade %{x}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))

    all_vals = list(p5) + list(p95) + list(actual)
    y_min, y_max = min(all_vals), max(all_vals)
    y_rng  = y_max - y_min
    padding = max(y_rng * 0.1, initial_equity * 0.005)

    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                   title=dict(text="Trade #", font=dict(size=11)), tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                   tickformat="$,.0f", tickfont=dict(size=11), title=None,
                   range=[y_min - padding, y_max + padding]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=10)),
        hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. P&L histogram
# ---------------------------------------------------------------------------

def _build_pnl_histogram(net_pnls: list[float], mode: str) -> go.Figure:
    fig = go.Figure()

    if mode == "all":
        fig.add_trace(go.Histogram(
            x=net_pnls, nbinsx=20,
            marker_color=_BLUE, opacity=0.85, name="All Trades",
        ))
    else:
        wins   = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]
        fig.add_trace(go.Histogram(
            x=wins, nbinsx=15,
            marker_color=_GREEN, opacity=0.75, name="Winners",
        ))
        fig.add_trace(go.Histogram(
            x=losses, nbinsx=15,
            marker_color=_RED, opacity=0.75, name="Losers",
        ))
        fig.update_layout(barmode="overlay")

    fig.add_vline(x=0, line_dash="dot", line_color=_GRAY, line_width=1)
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                   tickformat="$,.0f", tickfont=dict(size=11), title=None),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                   tickfont=dict(size=11),
                   title=dict(text="Count", font=dict(size=11))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        showlegend=(mode == "split"),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Metrics table
# ---------------------------------------------------------------------------

def _fmt(value: float, kind: str) -> str:
    if not math.isfinite(value):
        return "∞"
    if kind == "$":
        return f"${value:,.2f}"
    if kind == "%":
        return f"{value * 100:.2f}%"
    if kind == "x":
        return f"{value:.2f}×"
    if kind == "n":
        return f"{int(value):,}"
    return str(value)


def _build_metrics_table(m: dict) -> dag.AgGrid:
    rows = [
        {"_g": "trade",   "metric": "Net Profit",      "value": _fmt(m["net_profit"],          "$"), "_pnl": m["net_profit"]},
        {"_g": "trade",   "metric": "Total Trades",    "value": _fmt(m["total_trades"],         "n"), "_pnl": None},
        {"_g": "trade",   "metric": "Win Rate",        "value": _fmt(m["percent_profitable"],   "%"), "_pnl": None},
        {"_g": "trade",   "metric": "Profit Factor",   "value": _fmt(m["profit_factor"],        "x"), "_pnl": None},
        {"_g": "trade",   "metric": "Avg Win",         "value": _fmt(m["avg_win"],              "$"), "_pnl": None},
        {"_g": "trade",   "metric": "Avg Loss",        "value": _fmt(m["avg_loss"],             "$"), "_pnl": None},
        {"_g": "trade",   "metric": "Median P&L",      "value": _fmt(m["median_pnl"],          "$"), "_pnl": None},
        {"_g": "risk",    "metric": "Max Drawdown",    "value": _fmt(m["max_drawdown"],         "$"), "_pnl": None},
        {"_g": "risk",    "metric": "Max Drawdown %",  "value": _fmt(m["max_drawdown_pct"],     "%"), "_pnl": None},
        {"_g": "risk",    "metric": "CAGR",            "value": _fmt(m["cagr"],                 "%"), "_pnl": None},
        {"_g": "risk",    "metric": "Sharpe Ratio",    "value": _fmt(m["sharpe_ratio"],         "x"), "_pnl": None},
        {"_g": "risk",    "metric": "Sortino Ratio",   "value": _fmt(m["sortino_ratio"],        "x"), "_pnl": None},
        {"_g": "risk",    "metric": "Calmar Ratio",    "value": _fmt(m["calmar_ratio"],         "x"), "_pnl": None},
        {"_g": "quality", "metric": "Premium Capture", "value": _fmt(m["premium_capture_rate"], "%"), "_pnl": None},
        {"_g": "quality", "metric": "Avg MAE",         "value": _fmt(m["avg_mae"],              "$"), "_pnl": None},
        {"_g": "quality", "metric": "Worst MAE",       "value": _fmt(m["worst_mae"],            "$"), "_pnl": None},
    ]

    return dag.AgGrid(
        rowData=rows,
        columnDefs=[
            {
                "field": "metric", "headerName": "", "flex": 1,
                "cellStyle": {"color": _MUTED, "fontSize": "13px"},
            },
            {
                "field": "value", "headerName": "", "width": 110,
                "cellStyle": {
                    "textAlign": "right", "fontWeight": "600", "fontSize": "13px",
                    "styleConditions": [
                        {"condition": "params.data._pnl !== null && params.data._pnl >= 0",
                         "style": {"color": _GREEN}},
                        {"condition": "params.data._pnl !== null && params.data._pnl < 0",
                         "style": {"color": _RED}},
                    ],
                },
            },
        ],
        defaultColDef={"sortable": False, "filter": False, "resizable": False},
        dashGridOptions={
            "headerHeight": 0, "rowHeight": 28,
            "suppressMovableColumns": True,
        },
        style={"height": f"{len(rows) * 28 + 2}px"},
    )


# ---------------------------------------------------------------------------
# 5. Trade list
# ---------------------------------------------------------------------------

def _build_trade_table(positions: pl.DataFrame) -> dag.AgGrid:
    def _dur(minutes: float | None) -> str:
        if minutes is None:
            return "—"
        h, m = divmod(int(abs(minutes)), 60)
        return f"{h}h {m:02d}m" if h else f"{m}m"

    def _ts(dt) -> str:
        return dt.strftime("%m-%d %H:%M") if dt else "—"

    rows = []
    for i, r in enumerate(positions.to_dicts(), 1):
        ot, et   = r.get("open_time"), r.get("exit_time")
        dur_min  = (et - ot).total_seconds() / 60 if ot and et else None
        pnl      = r.get("net_pnl") or 0.0
        reason   = r.get("exit_reason", "")
        rows.append({
            "position_id": r["id"],
            "#":           i,
            "trade":       r.get("trade_name", ""),
            "open":        _ts(ot),
            "exit":        _ts(et),
            "duration":    _dur(dur_min),
            "reason":      reason,
            "open_pt":     round(r.get("open_mark", 0.0), 2),
            "exit_pt":     round(r.get("exit_mark",  0.0), 2),
            "pnl":         round(pnl, 2),
            "_reason":     reason,
        })

    reason_style = {
        "styleConditions": [
            {"condition": 'params.value === "take_profit"', "style": {"color": _GREEN}},
            {"condition": 'params.value === "stop_loss"',   "style": {"color": _RED}},
            {"condition": 'params.value === "expiry"',      "style": {"color": _AMBER}},
        ]
    }
    pnl_style = {
        "textAlign": "right",
        "styleConditions": [
            {"condition": "params.value > 0", "style": {"color": _GREEN, "fontWeight": "600"}},
            {"condition": "params.value < 0", "style": {"color": _RED,   "fontWeight": "600"}},
        ],
    }

    return dag.AgGrid(
        rowData=rows,
        columnDefs=[
            {"field": "position_id", "headerName": "", "width": 46,
             "cellRenderer": "ChartLink", "sortable": False, "filter": False,
             "pinned": "left"},
            {"field": "#",        "headerName": "#",        "width": 52,
             "cellStyle": {"color": _MUTED, "textAlign": "center"}},
            {"field": "trade",    "headerName": "Trade",    "width": 110},
            {"field": "open",     "headerName": "Open",     "width": 115},
            {"field": "exit",     "headerName": "Exit",     "width": 115},
            {"field": "duration", "headerName": "Dur.",     "width": 80,
             "cellStyle": {"textAlign": "center"}},
            {"field": "reason",   "headerName": "Reason",   "width": 110,
             "cellStyle": reason_style},
            {"field": "open_pt",  "headerName": "Open Pt",  "width": 88,
             "cellStyle": {"textAlign": "right"},
             "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
            {"field": "exit_pt",  "headerName": "Exit Pt",  "width": 88,
             "cellStyle": {"textAlign": "right"},
             "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
            {"field": "pnl",      "headerName": "Net P&L",  "width": 100,
             "cellStyle": pnl_style,
             "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"},
             "sort": "asc"},
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        dashGridOptions={
            "rowHeight": 32, "headerHeight": 36,
            "animateRows": True,
            "pagination": True, "paginationPageSize": 20,
            "suppressCellFocus": True,
        },
        style={"height": "520px"},
        className="ag-theme-alpine",
    )


# ---------------------------------------------------------------------------
# Trade chart page (Lightweight Charts JS, served via Flask route)
# ---------------------------------------------------------------------------

def _build_chart_html(
    position_id: int,
    output_db_path: str,
    input_db_path: str | None,
) -> str:
    from datetime import timezone

    with OutputDatabase(output_db_path) as odb:
        row = odb._con.execute(
            "SELECT trade_name, open_time, exit_time, exit_reason, "
            "open_mark, exit_mark, net_pnl FROM position WHERE id = ?",
            [position_id],
        ).fetchone()

    if row is None:
        return "<html><body><p>Position not found.</p></body></html>"

    trade_name, open_time, exit_time, exit_reason, open_mark, exit_mark, net_pnl = row

    open_str  = open_time.strftime("%m-%d %H:%M")  if open_time  else "—"
    exit_str  = exit_time.strftime("%m-%d %H:%M")  if exit_time  else "—"
    pnl_color = _GREEN if (net_pnl or 0) >= 0 else _RED
    pnl_str   = f"${net_pnl:+,.2f}" if net_pnl is not None else "—"

    candle_json  = "[]"
    markers_json = "[]"

    if input_db_path and open_time and exit_time:
        import duckdb
        day_start = open_time.replace(hour=0, minute=0, second=0, microsecond=0,
                                      tzinfo=timezone.utc)
        day_end   = exit_time.replace(hour=23, minute=59, second=59, microsecond=0,
                                      tzinfo=timezone.utc)

        con = duckdb.connect(input_db_path, read_only=True)
        try:
            bars = con.execute(
                """
                SELECT DISTINCT ON (ts_event) ts_event, open, high, low, close
                FROM underlying_bars
                WHERE ts_event >= ? AND ts_event <= ?
                ORDER BY ts_event
                """,
                [day_start, day_end],
            ).fetchall()
        finally:
            con.close()

        if bars:
            candles = [
                {"time": int(row[0].timestamp()),
                 "open": row[1], "high": row[2], "low": row[3], "close": row[4]}
                for row in bars
            ]
            candle_json = json.dumps(candles)

        reason_colors = {"take_profit": _GREEN, "stop_loss": _RED}
        m_color = reason_colors.get(exit_reason or "", _AMBER)
        markers = [
            {"time": int(open_time.timestamp()), "position": "belowBar",
             "color": _GREEN, "shape": "arrowUp",
             "text": f"Entry {open_mark:.2f}"},
            {"time": int(exit_time.timestamp()), "position": "aboveBar",
             "color": m_color, "shape": "arrowDown",
             "text": f"Exit {exit_mark:.2f} – {exit_reason}"},
        ]
        markers_json = json.dumps(markers)

    return _CHART_HTML.format(
        trade_name=trade_name or "—",
        open_str=open_str,
        exit_str=exit_str,
        exit_reason=exit_reason or "—",
        pnl_color=pnl_color,
        pnl_str=pnl_str,
        candle_json=candle_json,
        markers_json=markers_json,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_app(
    output_db_path: str,
    input_db_path: str | None = None,
    backtest_id: int | None = None,
) -> Dash:
    """Build and return the Dash app. All data loaded once at startup."""
    with OutputDatabase(output_db_path) as odb:
        pp             = PostProcessor(odb, backtest_id=backtest_id)
        bid            = pp._resolve_backtest_id()
        m              = pp.metrics()
        equity_df      = pp.equity_curve()
        initial_equity = pp._load_initial_equity()
        net_pnls       = pp._load_positions()["net_pnl"].to_list()

        # Load positions with DB id for chart links
        positions = odb._con.execute(
            "SELECT id, trade_name, open_time, exit_time, exit_reason, "
            "open_mark, exit_mark, net_pnl, worst_mark "
            "FROM position WHERE backtest_id = ? ORDER BY open_time",
            [bid],
        ).pl()

        meta = odb._con.execute(
            "SELECT strategy_name, created_at FROM backtest WHERE id = ?", [bid]
        ).fetchone()
        strategy_name = meta[0] if meta else "—"
        created_at    = meta[1].strftime("%Y-%m-%d %H:%M UTC") if (meta and meta[1]) else ""

    equity_fig   = _build_equity_chart(equity_df, initial_equity)
    boot_fig     = _build_bootstrap_fig(net_pnls, initial_equity)
    metrics_tbl  = _build_metrics_table(m)
    trade_tbl    = _build_trade_table(positions)

    app = Dash(
        __name__,
        assets_folder=_ASSETS,
        title=f"btkit — {strategy_name}",
    )

    # ── Flask route for per-trade Lightweight Charts page ───────────────
    server = app.server

    @server.route("/chart/<int:position_id>")
    def trade_chart_page(position_id: int):
        html_content = _build_chart_html(position_id, output_db_path, input_db_path)
        return Response(html_content, content_type="text/html; charset=utf-8")

    # ── Layout ───────────────────────────────────────────────────────────
    app.layout = html.Div(
        style={"fontFamily": "Inter, system-ui, sans-serif",
               "backgroundColor": _BG, "minHeight": "100vh"},
        children=[
            # Header
            html.Div(
                style={"backgroundColor": _HEADER, "color": "white",
                       "padding": "13px 24px", "display": "flex",
                       "alignItems": "center", "gap": "12px"},
                children=[
                    html.Span("btkit", style={"fontWeight": "700",
                              "fontSize": "17px", "letterSpacing": "0.04em"}),
                    html.Span("·", style={"color": "#4B5563", "fontSize": "18px"}),
                    html.Span(strategy_name, style={"fontSize": "14px", "color": "#D1D5DB"}),
                    html.Div(
                        style={"marginLeft": "auto", "display": "flex", "gap": "16px"},
                        children=[
                            html.Span(f"run #{bid}",  style={"fontSize": "12px", "color": "#9CA3AF"}),
                            html.Span(created_at,     style={"fontSize": "12px", "color": "#9CA3AF"}),
                        ],
                    ),
                ],
            ),

            # Body
            html.Div(
                style={"padding": "20px", "maxWidth": "1600px", "margin": "0 auto"},
                children=[

                    # Row 1: Equity curve + Bootstrap fan
                    html.Div(
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                        children=[
                            _card([
                                _section_label("Equity Curve"),
                                dcc.Graph(figure=equity_fig,
                                          config={"displayModeBar": False},
                                          style={"height": "280px"}),
                            ], style={"flex": "1"}),

                            _card([
                                _section_label(
                                    f"Bootstrap Equity Fan  ·  1 000 resamples  ·  "
                                    f"{len(net_pnls)} trades"
                                ),
                                dcc.Graph(figure=boot_fig,
                                          config={"displayModeBar": False},
                                          style={"height": "280px"}),
                            ], style={"flex": "1"}),
                        ],
                    ),

                    # Row 2: P&L Histogram (full width)
                    _card([
                        html.Div(
                            style={"display": "flex", "alignItems": "center",
                                   "marginBottom": "12px"},
                            children=[
                                _section_label("P&L Distribution"),
                                dcc.RadioItems(
                                    id="hist-mode",
                                    options=[
                                        {"label": " All trades",     "value": "all"},
                                        {"label": " Wins / Losses",  "value": "split"},
                                    ],
                                    value="all",
                                    inline=True,
                                    style={"marginLeft": "auto", "fontSize": "12px",
                                           "color": _MUTED, "gap": "12px"},
                                    inputStyle={"marginRight": "4px"},
                                ),
                            ],
                        ),
                        dcc.Graph(id="pnl-hist", config={"displayModeBar": False},
                                  style={"height": "220px"}),
                    ], style={"marginBottom": "16px"}),

                    # Row 3: Metrics + Trade list
                    html.Div(
                        style={"display": "flex", "gap": "16px",
                               "alignItems": "flex-start"},
                        children=[
                            _card([
                                _section_label("Strategy Metrics"),
                                metrics_tbl,
                            ], style={"width": "290px", "flexShrink": "0"}),

                            _card([
                                _section_label(
                                    f"All Trades  ·  {m['total_trades']} positions"
                                    + ("  ·  📈 click row chart icon to open trade chart" if input_db_path else "")
                                ),
                                trade_tbl,
                            ], style={"flex": "1", "minWidth": "0"}),
                        ],
                    ),
                ],
            ),
        ],
    )

    # ── Histogram callback ───────────────────────────────────────────────
    @app.callback(Output("pnl-hist", "figure"), Input("hist-mode", "value"))
    def update_histogram(mode: str) -> go.Figure:
        return _build_pnl_histogram(net_pnls, mode)

    return app


def run_dashboard(
    output_db_path: str,
    input_db_path: str | None = None,
    backtest_id: int | None = None,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Build the Dash app and start the development server."""
    app = create_app(output_db_path, input_db_path=input_db_path,
                     backtest_id=backtest_id)
    print(f"  Dashboard: http://localhost:{port}")
    app.run(port=port, debug=debug)
