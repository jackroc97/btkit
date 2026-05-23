"""
Dash dashboard for btkit backtest results.

Displays three components for a single backtest run:
  1. Equity curve — cumulative portfolio value over time
  2. Strategy metrics — key performance and risk statistics
  3. Trade list — every individual position with sortable/filterable columns

Launch via:
    btkit serve --output-db <path> [--backtest-id N] [--port 8050]
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import polars as pl
from dash import Dash, dash_table, dcc, html

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
        "fontSize": "11px",
        "fontWeight": "600",
        "letterSpacing": "0.08em",
        "textTransform": "uppercase",
        "color": _MUTED,
        "margin": "0 0 12px 0",
    })


# ---------------------------------------------------------------------------
# 1. Equity curve
# ---------------------------------------------------------------------------

def _build_equity_chart(equity_df: pl.DataFrame, initial_equity: float) -> go.Figure:
    if equity_df.is_empty():
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No trade data", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=_MUTED))],
            paper_bgcolor="white", plot_bgcolor="white",
        )
        return fig

    times    = equity_df["exit_time"].to_list()
    equities = equity_df["equity"].to_list()

    # Prepend the starting point so the curve begins at initial_equity
    times    = [times[0]] + times
    equities = [initial_equity] + equities

    final    = equities[-1]
    line_clr = _GREEN if final >= initial_equity else _RED
    fill_r, fill_g, fill_b = (
        int(line_clr[1:3], 16),
        int(line_clr[3:5], 16),
        int(line_clr[5:7], 16),
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=equities,
        mode="lines",
        line=dict(color=line_clr, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({fill_r},{fill_g},{fill_b},0.07)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))

    fig.add_hline(
        y=initial_equity,
        line_dash="dot",
        line_color=_GRAY,
        line_width=1,
        annotation_text=f"Start  ${initial_equity:,.0f}",
        annotation_font=dict(size=11, color=_GRAY),
        annotation_position="bottom right",
    )

    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            tickformat="%b %d", tickfont=dict(size=11),
            title=None, zeroline=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            tickformat="$,.0f", tickfont=dict(size=11),
            title=None, zeroline=False,
        ),
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Strategy metrics table
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


def _build_metrics_table(m: dict) -> dash_table.DataTable:
    rows = [
        {"metric": "Net Profit",        "value": _fmt(m["net_profit"],          "$")},
        {"metric": "Total Trades",      "value": _fmt(m["total_trades"],         "n")},
        {"metric": "Win Rate",          "value": _fmt(m["percent_profitable"],   "%")},
        {"metric": "Profit Factor",     "value": _fmt(m["profit_factor"],        "x")},
        {"metric": "Avg Win",           "value": _fmt(m["avg_win"],              "$")},
        {"metric": "Avg Loss",          "value": _fmt(m["avg_loss"],             "$")},
        {"metric": "Median P&L",        "value": _fmt(m["median_pnl"],          "$")},
        {"metric": "Max Drawdown",      "value": _fmt(m["max_drawdown"],         "$")},
        {"metric": "Max Drawdown %",    "value": _fmt(m["max_drawdown_pct"],     "%")},
        {"metric": "CAGR",              "value": _fmt(m["cagr"],                 "%")},
        {"metric": "Sharpe Ratio",      "value": _fmt(m["sharpe_ratio"],         "x")},
        {"metric": "Sortino Ratio",     "value": _fmt(m["sortino_ratio"],        "x")},
        {"metric": "Calmar Ratio",      "value": _fmt(m["calmar_ratio"],         "x")},
        {"metric": "Premium Capture",   "value": _fmt(m["premium_capture_rate"], "%")},
        {"metric": "Avg MAE",           "value": _fmt(m["avg_mae"],              "$")},
        {"metric": "Worst MAE",         "value": _fmt(m["worst_mae"],            "$")},
    ]

    # Divider row indices (0-based): after row 6 (end of trade stats), after row 12 (end of risk)
    divider_rows = [7, 13]

    profit_positive = m["net_profit"] >= 0

    conditional = [
        # Win/loss colour on Net Profit row
        {
            "if": {"row_index": 0, "column_id": "value"},
            "color": _GREEN if profit_positive else _RED,
            "fontWeight": "700",
        },
        # Alternating row tint
        *[
            {"if": {"row_index": i}, "backgroundColor": "#FAFAFA"}
            for i in range(0, len(rows) + len(divider_rows), 2)
        ],
    ]

    # Insert visual dividers
    for idx in sorted(divider_rows, reverse=True):
        rows.insert(idx, {"metric": "", "value": ""})

    conditional += [
        {
            "if": {"row_index": idx, "column_id": col},
            "borderTop": f"1px solid {_BORDER}",
            "padding": "2px 8px",
            "height": "4px",
        }
        for idx in divider_rows
        for col in ("metric", "value")
    ]

    return dash_table.DataTable(
        data=rows,
        columns=[
            {"name": "Metric", "id": "metric"},
            {"name": "Value",  "id": "value"},
        ],
        style_table={"overflowX": "hidden"},
        style_header={"display": "none"},
        style_cell={
            "fontFamily": "Inter, system-ui, sans-serif",
            "fontSize": "13px",
            "padding": "5px 8px",
            "border": "none",
            "color": _TEXT,
            "backgroundColor": "transparent",
        },
        style_cell_conditional=[
            {"if": {"column_id": "metric"}, "color": _MUTED, "width": "60%"},
            {"if": {"column_id": "value"},  "textAlign": "right", "fontWeight": "600"},
        ],
        style_data_conditional=conditional,
    )


# ---------------------------------------------------------------------------
# 3. Trade list table
# ---------------------------------------------------------------------------

def _build_trade_table(trades: pl.DataFrame) -> dash_table.DataTable:
    def _dur(minutes: float | None) -> str:
        if minutes is None:
            return "—"
        h, m = divmod(int(abs(minutes)), 60)
        return f"{h}h {m:02d}m" if h else f"{m}m"

    def _fmt_time(ts) -> str:
        if ts is None:
            return "—"
        return ts.strftime("%m-%d %H:%M")

    rows = []
    for i, r in enumerate(trades.to_dicts(), 1):
        dur_min = None
        ot, et = r.get("open_time"), r.get("exit_time")
        if ot and et:
            dur_min = (et - ot).total_seconds() / 60

        pnl = r.get("net_pnl") or 0.0
        rows.append({
            "#":          i,
            "Trade":      r.get("trade_name", ""),
            "Open":       _fmt_time(ot),
            "Exit":       _fmt_time(et),
            "Duration":   _dur(dur_min),
            "Reason":     r.get("exit_reason", ""),
            "Open Pt":    f"{r.get('open_mark', 0.0):.2f}",
            "Exit Pt":    f"{r.get('exit_mark', 0.0):.2f}",
            "Net P&L":    f"${pnl:,.2f}",
            "_pnl":       pnl,
            "_reason":    r.get("exit_reason", ""),
        })

    _cell_style = {
        "fontFamily": "Inter, system-ui, sans-serif",
        "fontSize": "12px",
        "padding": "6px 10px",
        "border": "none",
        "borderBottom": f"1px solid {_BORDER}",
        "color": _TEXT,
        "whiteSpace": "nowrap",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    }

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in
                 ["#", "Trade", "Open", "Exit", "Duration",
                  "Reason", "Open Pt", "Exit Pt", "Net P&L"]],
        sort_action="native",
        filter_action="native",
        page_size=25,
        style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "460px"},
        style_header={
            **_cell_style,
            "backgroundColor": "#F9FAFB",
            "fontWeight": "600",
            "fontSize": "11px",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
            "color": _MUTED,
            "borderBottom": f"2px solid {_BORDER}",
            "position": "sticky",
            "top": "0",
        },
        style_cell=_cell_style,
        style_cell_conditional=[
            {"if": {"column_id": "#"},        "width": "40px",  "textAlign": "center", "color": _MUTED},
            {"if": {"column_id": "Trade"},    "width": "100px"},
            {"if": {"column_id": "Open"},     "width": "110px"},
            {"if": {"column_id": "Exit"},     "width": "110px"},
            {"if": {"column_id": "Duration"}, "width": "80px",  "textAlign": "center"},
            {"if": {"column_id": "Reason"},   "width": "100px"},
            {"if": {"column_id": "Open Pt"},  "width": "80px",  "textAlign": "right"},
            {"if": {"column_id": "Exit Pt"},  "width": "80px",  "textAlign": "right"},
            {"if": {"column_id": "Net P&L"},  "width": "90px",  "textAlign": "right"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#FAFAFA"},
            # P&L colouring
            {"if": {"filter_query": "{_pnl} > 0", "column_id": "Net P&L"},
             "color": _GREEN, "fontWeight": "600"},
            {"if": {"filter_query": "{_pnl} < 0", "column_id": "Net P&L"},
             "color": _RED, "fontWeight": "600"},
            # Exit reason colouring
            {"if": {"filter_query": '{_reason} = "take_profit"', "column_id": "Reason"},
             "color": _GREEN},
            {"if": {"filter_query": '{_reason} = "stop_loss"',  "column_id": "Reason"},
             "color": _RED},
            {"if": {"filter_query": '{_reason} = "expiry"',     "column_id": "Reason"},
             "color": _AMBER},
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_app(output_db_path: str, backtest_id: int | None = None) -> Dash:
    """
    Build and return the Dash app. All data is loaded once at startup;
    the database connection is closed before the server begins accepting requests.
    """
    with OutputDatabase(output_db_path) as odb:
        pp             = PostProcessor(odb, backtest_id=backtest_id)
        bid            = pp._resolve_backtest_id()
        m              = pp.metrics()
        equity_df      = pp.equity_curve()
        trades         = pp.trade_pnl_series()
        initial_equity = pp._load_initial_equity()

        row = odb._con.execute(
            "SELECT strategy_name, created_at FROM backtest WHERE id = ?", [bid]
        ).fetchone()
        strategy_name = row[0] if row else "—"
        created_at    = row[1].strftime("%Y-%m-%d %H:%M UTC") if (row and row[1]) else ""

    equity_fig   = _build_equity_chart(equity_df, initial_equity)
    metrics_tbl  = _build_metrics_table(m)
    trade_tbl    = _build_trade_table(trades)

    app = Dash(__name__, title=f"btkit — {strategy_name}")

    app.layout = html.Div(
        style={
            "fontFamily": "Inter, system-ui, sans-serif",
            "backgroundColor": _BG,
            "minHeight": "100vh",
        },
        children=[
            # ── Header bar ──────────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": _HEADER,
                    "color": "white",
                    "padding": "13px 24px",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                },
                children=[
                    html.Span("btkit", style={
                        "fontWeight": "700", "fontSize": "17px",
                        "letterSpacing": "0.04em",
                    }),
                    html.Span("·", style={"color": "#4B5563", "fontSize": "18px"}),
                    html.Span(strategy_name, style={
                        "fontSize": "14px", "color": "#D1D5DB",
                    }),
                    html.Div(style={"marginLeft": "auto", "display": "flex", "gap": "16px"},
                             children=[
                        html.Span(f"run #{bid}", style={"fontSize": "12px", "color": "#9CA3AF"}),
                        html.Span(created_at,    style={"fontSize": "12px", "color": "#9CA3AF"}),
                    ]),
                ],
            ),

            # ── Body ────────────────────────────────────────────────────
            html.Div(
                style={"padding": "20px", "maxWidth": "1600px", "margin": "0 auto"},
                children=[

                    # Equity curve
                    _card([
                        _section_label("Equity Curve"),
                        dcc.Graph(
                            figure=equity_fig,
                            config={"displayModeBar": False},
                            style={"height": "300px"},
                        ),
                    ], style={"marginBottom": "16px"}),

                    # Metrics + trade list side by side
                    html.Div(
                        style={"display": "flex", "gap": "16px", "alignItems": "flex-start"},
                        children=[
                            _card([
                                _section_label("Strategy Metrics"),
                                metrics_tbl,
                            ], style={"width": "290px", "flexShrink": "0"}),

                            _card([
                                _section_label(
                                    f"All Trades  ·  {m['total_trades']} positions"
                                ),
                                trade_tbl,
                            ], style={"flex": "1", "minWidth": "0"}),
                        ],
                    ),
                ],
            ),
        ],
    )

    return app


def run_dashboard(
    output_db_path: str,
    backtest_id: int | None = None,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Build the Dash app and start the development server."""
    app = create_app(output_db_path, backtest_id=backtest_id)
    print(f"  Dashboard: http://localhost:{port}")
    app.run(port=port, debug=debug)
