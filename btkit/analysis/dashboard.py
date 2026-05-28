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
from datetime import UTC
from pathlib import Path

import dash_ag_grid as dag
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, ctx, dcc, html
from flask import Response

from btkit.analysis.metrics import PostProcessor
from btkit.db.output_db import OutputDatabase

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BLUE = "#2563EB"
_GREEN = "#16A34A"
_RED = "#DC2626"
_AMBER = "#D97706"
_GRAY = "#6B7280"
_BG = "#F3F4F6"
_CARD = "#FFFFFF"
_BORDER = "#E5E7EB"
_TEXT = "#111827"
_MUTED = "#6B7280"
_HEADER = "#111827"

_ASSETS = str(Path(__file__).parent / "assets")

# Categorical palette for multi-curve study equity chart (cycles for >10 traces)
_STUDY_PALETTE = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#DB2777", "#0891B2", "#65A30D", "#EA580C", "#6366F1",
]

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
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: Inter, system-ui, sans-serif; background: #fff;
      display: flex; flex-direction: column; height: 100vh; overflow: hidden;
    }}
    #hdr {{
      background: #111827; color: #fff; padding: 0 16px;
      font-size: 13px; display: flex; align-items: center; gap: 14px;
      height: 42px; flex-shrink: 0;
    }}
    #hdr b       {{ font-size: 15px; letter-spacing: .04em; }}
    #hdr .sep    {{ color: #4B5563; }}
    #hdr .muted  {{ color: #9CA3AF; font-size: 12px; }}
    #hdr .pnl    {{ margin-left: auto; font-weight: 700; font-size: 14px; }}
    #hdr .reason {{ color: #9CA3AF; font-size: 12px; }}
    #chart-wrap  {{ flex: 65; min-height: 0; position: relative; }}
    #chart       {{ position: absolute; inset: 0; }}
    #pnl-label   {{
      height: 22px; flex-shrink: 0;
      background: #F9FAFB; border-top: 1px solid #E5E7EB;
      padding: 3px 10px; font-size: 11px; font-weight: 600;
      letter-spacing: .06em; text-transform: uppercase; color: #6B7280;
    }}
    #pnl-wrap    {{ flex: 35; min-height: 0; position: relative; }}
    #pnl-chart   {{ position: absolute; inset: 0; }}
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
<div id="chart-wrap"><div id="chart"></div></div>
<div id="pnl-label">Running P&amp;L</div>
<div id="pnl-wrap"><div id="pnl-chart"></div></div>
<script>
const candleData    = {candle_json};
const volumeData    = {volume_json};
const markers       = {markers_json};
const beLines       = {be_lines_json};
const pnlData       = {pnl_json};
const afterExitData = {after_exit_json};
const tpSlLines     = {tp_sl_json};
const strikeLines   = {strike_lines_json};
const openTs        = {open_ts};
const exitTs        = {exit_ts};

// ── Timezone helpers (America/New_York) ─────────────────────────────────────
const _etTime = ts => new Date(ts * 1000).toLocaleTimeString('en-US', {{
  timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false,
}});
const _etDate = ts => new Date(ts * 1000).toLocaleDateString('en-US', {{
  timeZone: 'America/New_York', month: 'short', day: 'numeric',
}});
const _etDateTime = ts => `${{_etDate(ts)}} ${{_etTime(ts)}}`;
const _etTickFmt = (ts, type) => type < 3 ? _etDate(ts) : _etTime(ts);

// ── Main candlestick + volume chart ─────────────────────────────────────────
const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  layout: {{ background: {{ color: '#fff' }}, textColor: '#374151' }},
  grid:   {{ vertLines: {{ color: '#F3F4F6' }}, horzLines: {{ color: '#F3F4F6' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: '#E5E7EB' }},
  localization: {{ timeFormatter: _etDateTime }},
  timeScale: {{
    borderColor: '#E5E7EB', timeVisible: true, secondsVisible: false,
    tickMarkFormatter: _etTickFmt,
  }},
  width:  document.getElementById('chart').clientWidth,
  height: document.getElementById('chart').clientHeight,
}});

const cs = chart.addCandlestickSeries({{
  upColor: '#16A34A', downColor: '#DC2626',
  borderUpColor: '#16A34A', borderDownColor: '#DC2626',
  wickUpColor: '#16A34A', wickDownColor: '#DC2626',
}});

const volSeries = chart.addHistogramSeries({{
  priceFormat: {{ type: 'volume' }},
  priceScaleId: 'volume',
}});
volSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.82, bottom: 0 }} }});

if (candleData.length > 0) {{
  cs.setData(candleData);
  cs.setMarkers(markers);
  if (volumeData.length > 0) volSeries.setData(volumeData);

  for (const be of beLines) {{
    cs.createPriceLine({{
      price: be.price,
      color: '#6BBCED',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true,
      title: be.label,
    }});
  }}

  for (const sl of strikeLines) {{
    cs.createPriceLine({{
      price: sl.price,
      color: '#9CA3AF',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.LargeDashed,
      axisLabelVisible: true,
      title: sl.label,
    }});
  }}

  chart.timeScale().fitContent();
}} else {{
  chart.applyOptions({{
    watermark: {{
      visible: true, fontSize: 18, horzAlign: 'center', vertAlign: 'center',
      color: '#9CA3AF', text: 'No underlying bar data (pass --input-db to btkit serve)',
    }},
  }});
}}

// ── Running P&L chart ───────────────────────────────────────────────────────
const pnlChart = LightweightCharts.createChart(document.getElementById('pnl-chart'), {{
  layout: {{ background: {{ color: '#F9FAFB' }}, textColor: '#374151' }},
  grid:   {{ vertLines: {{ color: '#E5E7EB' }}, horzLines: {{ color: '#E5E7EB' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: '#E5E7EB', autoScale: true }},
  localization: {{ timeFormatter: _etDateTime }},
  timeScale: {{ borderColor: '#E5E7EB', timeVisible: true, secondsVisible: false, visible: false }},
  handleScroll: false,
  handleScale:  false,
  width:  document.getElementById('pnl-chart').clientWidth,
  height: document.getElementById('pnl-chart').clientHeight,
}});

const pnlSeries = pnlChart.addBaselineSeries({{
  baseValue: {{ type: 'price', price: 0 }},
  topLineColor:     '#16A34A',
  topFillColor1:    'rgba(22,163,74,0.15)',
  topFillColor2:    'rgba(22,163,74,0.02)',
  bottomLineColor:  '#DC2626',
  bottomFillColor1: 'rgba(220,38,38,0.02)',
  bottomFillColor2: 'rgba(220,38,38,0.15)',
  lineWidth: 1,
  priceFormat: {{ type: 'price', precision: 2, minMove: 0.01 }},
}});

if (pnlData.length > 0) {{
  // Map every candle timestamp into the P&L series. The series must span the full
  // candle time domain so both charts share the same logical time axis and scroll
  // 1:1 at any zoom level. Within the trade window forward-fill the last known
  // value to eliminate gaps from sparse option bar ticks. Outside the trade window
  // emit null so no line is drawn, but the time domain anchor still exists.
  const pnlMap = {{}};
  pnlData.forEach(d => {{ pnlMap[d.time] = d.value; }});
  let _lastPnl = null;
  const paddedPnl = candleData.map(c => {{
    if (pnlMap[c.time] !== undefined) _lastPnl = pnlMap[c.time];
    const inTrade = c.time >= openTs && c.time <= exitTs;
    return {{ time: c.time, value: (inTrade && _lastPnl !== null) ? _lastPnl : null }};
  }});
  pnlSeries.setData(paddedPnl);

  // Gray dashed line: hypothetical P&L if position had been held after exit
  if (afterExitData.length > 1) {{
    const afterSeries = pnlChart.addLineSeries({{
      color: '#9CA3AF',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      priceFormat: {{ type: 'price', precision: 2, minMove: 0.01 }},
      lastValueVisible: false,
      priceLineVisible: false,
    }});
    afterSeries.setData(afterExitData);
  }}

  // TP / SL horizontal lines on P&L pane
  for (const line of tpSlLines) {{
    pnlSeries.createPriceLine({{
      price: line.value,
      color: line.color,
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true,
      title: line.label,
    }});
  }}
}}

// ── Shared initial view: trade open → exit with padding ─────────────────────
const pad = Math.max((exitTs - openTs) * 0.08, 900);
const viewRange = {{ from: openTs - pad, to: exitTs + pad }};
if (candleData.length > 0) chart.timeScale().setVisibleRange(viewRange);
if (pnlData.length   > 0) pnlChart.timeScale().setVisibleRange(viewRange);

// ── Sync: candle drives P&L (P&L has handleScroll/Scale disabled) ───────────
chart.timeScale().subscribeVisibleTimeRangeChange(range => {{
  if (range !== null) pnlChart.timeScale().setVisibleRange(range);
}});

// ── Resize ──────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  chart.applyOptions({{
    width:  document.getElementById('chart').clientWidth,
    height: document.getElementById('chart').clientHeight,
  }});
  pnlChart.applyOptions({{
    width:  document.getElementById('pnl-chart').clientWidth,
    height: document.getElementById('pnl-chart').clientHeight,
  }});
}});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _card(children, style: dict | None = None, className: str = "") -> html.Div:
    base: dict = {
        "backgroundColor": _CARD,
        "border": f"1px solid {_BORDER}",
        "borderRadius": "8px",
        "padding": "18px 20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.06)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base, className=f"btkit-card {className}".strip())


def _section_label(text: str) -> html.P:
    return html.P(
        text,
        style={
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "0.08em",
            "textTransform": "uppercase",
            "color": _MUTED,
            "margin": "0 0 12px 0",
        },
    )


# ---------------------------------------------------------------------------
# 1. Equity curve
# ---------------------------------------------------------------------------


def _build_equity_chart(equity_df: pl.DataFrame, initial_equity: float) -> go.Figure:
    if equity_df.is_empty():
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(
                    text="No trade data",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color=_MUTED),
                )
            ],
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    times = equity_df["exit_time"].to_list()
    equities = equity_df["equity"].to_list()
    times = [times[0]] + times
    pnls = [e - initial_equity for e in [initial_equity] + equities]

    pnl_min = min(pnls)
    pnl_max = max(pnls)
    padding = max((pnl_max - pnl_min) * 0.25, abs(pnl_max - pnl_min) * 0.1 + 1)

    # Split series at zero crossings so each segment is colored independently
    def _zero_segments(ts, vs):
        segs: list[tuple[list, list]] = []
        if not ts:
            return segs
        seg_t, seg_v = [ts[0]], [vs[0]]
        for i in range(1, len(ts)):
            t0, v0, t1, v1 = ts[i - 1], vs[i - 1], ts[i], vs[i]
            if v0 * v1 < 0:  # sign change — interpolate crossing point
                alpha = v0 / (v0 - v1)
                t_cross = t0 + (t1 - t0) * alpha
                seg_t.append(t_cross)
                seg_v.append(0.0)
                segs.append((list(seg_t), list(seg_v)))
                seg_t, seg_v = [t_cross, t1], [0.0, v1]
            else:
                seg_t.append(t1)
                seg_v.append(v1)
        segs.append((seg_t, seg_v))
        return segs

    fig = go.Figure()
    first = True
    for seg_t, seg_v in _zero_segments(times, pnls):
        is_pos = any(v > 0 for v in seg_v)
        clr = _GREEN if is_pos else _RED
        r, g, b = int(clr[1:3], 16), int(clr[3:5], 16), int(clr[5:7], 16)
        fig.add_trace(
            go.Scatter(
                x=seg_t,
                y=seg_v,
                mode="lines",
                line=dict(color=clr, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({r},{g},{b},0.12)",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>$%{y:+,.2f}</b><extra></extra>"
                if first
                else None,
                showlegend=False,
            )
        )
        first = False

    fig.add_hline(y=0, line_dash="dot", line_color=_GRAY, line_width=1)
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            tickformat="%b %d",
            tickfont=dict(size=11),
            title=None,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            tickformat="$+,.0f",
            tickfont=dict(size=11),
            title=None,
            zeroline=False,
            range=[pnl_min - padding, pnl_max + padding],
        ),
        showlegend=False,
        hovermode="x unified",
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
    x = list(range(n + 1))

    rng = np.random.default_rng(42)
    samples = rng.choice(arr, size=(n_iter, n), replace=True)
    cum = np.hstack([np.zeros((n_iter, 1)), np.cumsum(samples, axis=1)])

    p5, p10 = np.percentile(cum, 5, axis=0), np.percentile(cum, 10, axis=0)
    p25, p50 = np.percentile(cum, 25, axis=0), np.percentile(cum, 50, axis=0)
    p75, p90 = np.percentile(cum, 75, axis=0), np.percentile(cum, 90, axis=0)
    p95 = np.percentile(cum, 95, axis=0)

    actual = np.concatenate([[0], np.cumsum(arr)])
    line_clr = _GREEN if actual[-1] >= 0 else _RED

    fig = go.Figure()

    def _band(lo, hi, fill_alpha, name):
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(hi) + list(lo[::-1]),
                fill="toself",
                fillcolor=f"rgba(37,99,235,{fill_alpha})",
                line=dict(width=0),
                name=name,
                showlegend=True,
                hoverinfo="none",
            )
        )

    _band(p5, p95, 0.06, "5th–95th %ile")
    _band(p10, p90, 0.08, "10th–90th %ile")
    _band(p25, p75, 0.12, "25th–75th %ile")

    fig.add_trace(
        go.Scatter(
            x=x,
            y=list(p50),
            mode="lines",
            line=dict(color=_BLUE, width=1.5, dash="dash"),
            name="Median",
            showlegend=True,
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=list(actual),
            mode="lines",
            line=dict(color=line_clr, width=2.5),
            name="Actual",
            showlegend=True,
            hovertemplate="Trade %{x}<br><b>$%{y:+,.2f}</b><extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color=_GRAY, line_width=1)

    all_vals = list(p5) + list(p95) + list(actual)
    y_min, y_max = min(all_vals), max(all_vals)
    y_rng = y_max - y_min
    padding = max(y_rng * 0.1, abs(y_rng) * 0.05 + 1)

    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            title=dict(text="Trade #", font=dict(size=11)),
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            tickformat="$+,.0f",
            tickfont=dict(size=11),
            title=None,
            range=[y_min - padding, y_max + padding],
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)
        ),
        hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. P&L histogram
# ---------------------------------------------------------------------------


def _build_pnl_histogram(net_pnls: list[float], mode: str) -> go.Figure:
    fig = go.Figure()

    if mode == "winners":
        data = [p for p in net_pnls if p > 0]
        color, name = _GREEN, "Winners"
    elif mode == "losers":
        data = [p for p in net_pnls if p <= 0]
        color, name = _RED, "Losers"
    else:
        data, color, name = net_pnls, _BLUE, "All Trades"

    if data:
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=20,
                marker_color=color,
                opacity=0.85,
                name=name,
            )
        )

    fig.add_vline(x=0, line_dash="dot", line_color=_GRAY, line_width=1)
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            tickformat="$,.0f",
            tickfont=dict(size=11),
            title=None,
            autorange=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            tickfont=dict(size=11),
            title=dict(text="Count", font=dict(size=11)),
            autorange=True,
        ),
        showlegend=False,
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


def _kv_list(rows: list[tuple[str, str, dict | None]]) -> html.Div:
    """
    Shared key-value list renderer used by both params and metrics panels.
    rows: list of (label, value, optional value_style_overrides)
    """
    _row_style = {
        "display": "flex",
        "justifyContent": "space-between",
        "padding": "3px 0",
        "borderBottom": f"1px solid {_BORDER}",
        "fontSize": "12px",
        "lineHeight": "1.5",
    }
    _label_style = {"color": _MUTED, "flexShrink": "0", "paddingRight": "8px"}
    _value_base = {"fontFamily": "monospace", "textAlign": "right", "wordBreak": "break-all"}

    items = []
    for label, value, extra_style in rows:
        val_style = {**_value_base, **(extra_style or {})}
        items.append(html.Div([
            html.Span(label, style=_label_style),
            html.Span(value, style=val_style),
        ], style=_row_style))
    return html.Div(items)


def _build_params_panel(strategy_params: dict) -> html.Div:
    """
    Compact key-value table of strategy parameters: universe dates, costs,
    and per-trade entry/leg/exit settings. Mirrors the width of the metrics card.
    """
    rows: list[tuple[str, str]] = []

    uni = strategy_params.get("universe", {})
    rows.append(("Start date", uni.get("start_date", "—")))
    rows.append(("End date",   uni.get("end_date",   "—")))

    costs = strategy_params.get("costs", {})
    if costs.get("fee_per_contract"):
        rows.append(("Fee / contract", f"${costs['fee_per_contract']:g}"))
    if costs.get("slippage_pct"):
        rows.append(("Slippage", f"{costs['slippage_pct'] * 100:g}%"))

    for trade in strategy_params.get("trades", []):
        tn = trade.get("name", "trade")
        prefix = f"{tn}." if len(strategy_params.get("trades", [])) > 1 else ""

        entry = trade.get("entry", {})
        win = entry.get("window", {})
        if win:
            rows.append((f"{prefix}entry window", f"{win.get('start','?')}–{win.get('end','?')}"))
        for cond in entry.get("conditions", []):
            rows.append((f"{prefix}entry cond", cond))
        if entry.get("min_credit") is not None:
            rows.append((f"{prefix}min credit", f"${entry['min_credit']:g}"))
        if entry.get("max_debit") is not None:
            rows.append((f"{prefix}max debit",  f"${entry['max_debit']:g}"))

        for leg in trade.get("legs", []):
            ln = leg["name"]
            action_short = "sell" if "sell" in (leg.get("action") or "") else "buy"
            right = leg.get("right", "?")
            dte = leg.get("dte")
            delta = leg.get("delta")
            strike_offset = leg.get("strike_offset")
            ref = leg.get("reference_leg")
            qty = leg.get("quantity", 1)

            desc_parts = [f"{action_short} {right}"]
            if delta is not None:
                desc_parts.append(f"δ {delta:g}")
            if strike_offset is not None:
                desc_parts.append(f"offset {strike_offset:+g}")
            if ref:
                desc_parts.append(f"ref:{ref}")
            if strike_offset is not None:
                desc_parts.append("inherited" if dte is None else f"{dte}d")
            elif dte is not None:
                desc_parts.append(f"{dte}d")
            if qty != 1:
                desc_parts.append(f"×{qty}")
            rows.append((f"{prefix}{ln}", "  ".join(desc_parts)))

        ex = trade.get("exit", {})
        sl = ex.get("stop_loss")
        if sl is not None:
            if isinstance(sl, dict):
                price = sl.get("price") or sl.get("pct")
                cond = sl.get("condition")
                sl_str = f"{price:g}" if isinstance(price, (int, float)) else str(price)
                if cond:
                    sl_str += f"  [{cond}]"
            else:
                sl_str = f"{sl:g}" if isinstance(sl, (int, float)) else str(sl)
            rows.append((f"{prefix}stop loss", sl_str))

        tp = ex.get("take_profit")
        if tp is not None:
            if isinstance(tp, dict):
                price = tp.get("price")
                pct = tp.get("pct")
                cond = tp.get("condition")
                if pct is not None:
                    tp_str = f"{float(pct) * 100:g}% of credit"
                elif price is not None:
                    tp_str = f"{price:g}"
                else:
                    tp_str = "—"
                if cond:
                    tp_str += f"  [{cond}]"
            else:
                tp_str = f"{tp:g}" if isinstance(tp, (int, float)) else str(tp)
            rows.append((f"{prefix}take profit", tp_str))

        tp_pct = ex.get("take_profit_pct")
        if tp_pct is not None:
            rows.append((f"{prefix}take profit", f"{float(tp_pct) * 100:g}% of credit"))

        dte_exit = ex.get("dte_exit")
        if dte_exit is not None:
            rows.append((f"{prefix}DTE exit", f"{dte_exit}d"))

        for cond in ex.get("conditions", []):
            rows.append((f"{prefix}exit cond", cond))

    return html.Div([
        _section_label("Strategy Parameters"),
        _kv_list([(label, value, None) for label, value in rows]),
    ])


def _build_metrics_table(m: dict) -> html.Div:
    net = m["net_profit"]
    pnl_color = _GREEN if net >= 0 else _RED

    rows: list[tuple[str, str, dict | None]] = [
        ("Net Profit",       _fmt(net,                      "$"), {"color": pnl_color, "fontWeight": "600"}),
        ("Total Trades",     _fmt(m["total_trades"],        "n"), None),
        ("Win Rate",         _fmt(m["percent_profitable"],  "%"), None),
        ("Profit Factor",    _fmt(m["profit_factor"],       "x"), None),
        ("Avg Win",          _fmt(m["avg_win"],             "$"), None),
        ("Avg Loss",         _fmt(m["avg_loss"],            "$"), None),
        ("Median P&L",       _fmt(m["median_pnl"],          "$"), None),
        ("Max Drawdown",     _fmt(m["max_drawdown"],        "$"), None),
        ("Max Drawdown %",   _fmt(m["max_drawdown_pct"],    "%"), None),
        ("CAGR",             _fmt(m["cagr"],                "%"), None),
        ("Sharpe Ratio",     _fmt(m["sharpe_ratio"],        "x"), None),
        ("Sortino Ratio",    _fmt(m["sortino_ratio"],       "x"), None),
        ("Calmar Ratio",     _fmt(m["calmar_ratio"],        "x"), None),
        ("Premium Capture",  _fmt(m["premium_capture_rate"],"%"), None),
        ("Avg MAE",          _fmt(m["avg_mae"],             "$"), None),
        ("Worst MAE",        _fmt(m["worst_mae"],           "$"), None),
    ]

    return html.Div([
        _section_label("Strategy Metrics"),
        _kv_list(rows),
    ])


# ---------------------------------------------------------------------------
# 5. Trade list
# ---------------------------------------------------------------------------


def _build_trade_table(
    positions: pl.DataFrame,
    leg_strings: dict | None = None,
    underlying_symbols: dict | None = None,
) -> dag.AgGrid:
    def _dur(minutes: float | None) -> str:
        if minutes is None:
            return "—"
        h, m = divmod(int(abs(minutes)), 60)
        return f"{h:02d}:{m:02d}"

    def _ts(dt) -> str:
        return dt.strftime("%m-%d %H:%M") if dt else "—"

    leg_strings = leg_strings or {}
    underlying_symbols = underlying_symbols or {}
    rows = []
    for i, r in enumerate(positions.to_dicts(), 1):
        ot, et = r.get("open_time"), r.get("exit_time")
        dur_min = (et - ot).total_seconds() / 60 if ot and et else None
        pnl = r.get("net_pnl") or 0.0
        reason = r.get("exit_reason", "")
        pid = r["id"]
        rows.append(
            {
                "position_id": pid,
                "#": i,
                "trade": r.get("trade_name", ""),
                "underlying": underlying_symbols.get(pid, ""),
                "contracts": leg_strings.get(pid, ""),
                "open": _ts(ot),
                "exit": _ts(et),
                "duration": _dur(dur_min),
                "reason": reason,
                "open_pt": round(r.get("open_mark", 0.0), 2),
                "exit_pt": round(r.get("exit_mark", 0.0), 2),
                "pnl": round(pnl, 2),
                "_reason": reason,
            }
        )

    reason_style = {
        "styleConditions": [
            {"condition": 'params.value === "take_profit"', "style": {"color": _GREEN}},
            {"condition": 'params.value === "stop_loss"', "style": {"color": _RED}},
            {"condition": 'params.value === "expiry"', "style": {"color": _AMBER}},
        ]
    }
    pnl_style = {
        "textAlign": "right",
        "styleConditions": [
            {"condition": "params.value > 0", "style": {"color": _GREEN, "fontWeight": "600"}},
            {"condition": "params.value < 0", "style": {"color": _RED, "fontWeight": "600"}},
        ],
    }

    return dag.AgGrid(
        rowData=rows,
        columnDefs=[
            {
                "field": "position_id",
                "headerName": "",
                "width": 46,
                "cellRenderer": "ChartLink",
                "sortable": False,
                "filter": False,
                "pinned": "left",
                "suppressSizeToFit": True,
            },
            {
                "field": "#",
                "headerName": "#",
                "width": 52,
                "cellStyle": {"color": _MUTED, "textAlign": "center"},
                "suppressSizeToFit": True,
            },
            {"field": "trade", "headerName": "Trade", "minWidth": 90},
            {
                "field": "underlying",
                "headerName": "Underlying",
                "minWidth": 80,
                "cellStyle": {"fontFamily": "monospace", "fontSize": "12px"},
            },
            {
                "field": "contracts",
                "headerName": "Contracts",
                "minWidth": 200,
                "cellStyle": {
                    "fontFamily": "monospace",
                    "fontSize": "12px",
                    "color": _MUTED,
                    "letterSpacing": "0.02em",
                },
            },
            {"field": "open", "headerName": "Open", "minWidth": 105},
            {"field": "exit", "headerName": "Exit", "minWidth": 105},
            {
                "field": "duration",
                "headerName": "Dur.",
                "minWidth": 70,
                "cellStyle": {"textAlign": "center"},
            },
            {"field": "reason", "headerName": "Reason", "minWidth": 100, "cellStyle": reason_style},
            {
                "field": "open_pt",
                "headerName": "Open Pt",
                "minWidth": 80,
                "cellStyle": {"textAlign": "right"},
                "valueFormatter": {"function": "d3.format(',.2f')(params.value)"},
            },
            {
                "field": "exit_pt",
                "headerName": "Exit Pt",
                "minWidth": 80,
                "cellStyle": {"textAlign": "right"},
                "valueFormatter": {"function": "d3.format(',.2f')(params.value)"},
            },
            {
                "field": "pnl",
                "headerName": "Net P&L",
                "minWidth": 95,
                "cellStyle": pnl_style,
                "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"},
                "sort": "asc",
            },
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True, "flex": 1},
        dashGridOptions={
            "rowHeight": 32,
            "headerHeight": 36,
            "animateRows": True,
            "pagination": True,
            "paginationPageSize": 20,
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
    from itertools import groupby as _groupby

    with OutputDatabase(output_db_path) as odb:
        row = odb._con.execute(
            "SELECT trade_name, open_time, exit_time, exit_reason, "
            "open_mark, exit_mark, net_pnl, backtest_id FROM position WHERE id = ?",
            [position_id],
        ).fetchone()
        legs = odb._con.execute(
            'SELECT instrument_id, action, quantity, multiplier, open_price, "right", strike_price '
            "FROM position_leg WHERE position_id = ?",
            [position_id],
        ).fetchall()
        strategy_row = None
        if row is not None:
            strategy_row = odb._con.execute(
                "SELECT strategy_params FROM backtest WHERE id = ?", [row[7]]
            ).fetchone()

    if row is None:
        return "<html><body><p>Position not found.</p></body></html>"

    trade_name, open_time, exit_time, exit_reason, open_mark, exit_mark, net_pnl, _bid = row

    # Parse strategy TP/SL params
    import json as _json

    tp_pct: float | None = None
    tp_abs: float | None = None
    sl_pts: float | None = None
    if strategy_row and strategy_row[0]:
        try:
            sp = _json.loads(strategy_row[0])
            trades = sp.get("trades", [])
            if trades:
                exit_cfg = trades[0].get("exit", {})
                tp_pct = exit_cfg.get("take_profit_pct")
                tp_abs = exit_cfg.get("take_profit")
                sl_pts = exit_cfg.get("stop_loss")
        except Exception:
            pass

    open_str = open_time.strftime("%m-%d %H:%M") if open_time else "—"
    exit_str = exit_time.strftime("%m-%d %H:%M") if exit_time else "—"
    pnl_color = _GREEN if (net_pnl or 0) >= 0 else _RED
    pnl_str = f"${net_pnl:+,.2f}" if net_pnl is not None else "—"

    candle_json = "[]"
    volume_json = "[]"
    markers_json = "[]"
    be_lines_json = "[]"
    pnl_json = "[]"
    after_exit_json = "[]"
    tp_sl_json = "[]"
    strike_lines_json = "[]"
    open_ts = int(open_time.timestamp()) if open_time else 0
    exit_ts = int(exit_time.timestamp()) if exit_time else 0

    if input_db_path and open_time and exit_time:
        import duckdb

        day_start = open_time.replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
        )
        day_end = exit_time.replace(
            hour=23, minute=59, second=59, microsecond=0, tzinfo=UTC
        )

        con = duckdb.connect(input_db_path, read_only=True)
        try:
            # Resolve underlying instrument from the first leg
            underlying_id = None
            if legs:
                uid_row = con.execute(
                    "SELECT underlying_id FROM option_bars WHERE instrument_id = ? LIMIT 1",
                    [legs[0][0]],
                ).fetchone()
                if uid_row:
                    underlying_id = uid_row[0]

            # Underlying OHLCV
            if underlying_id is not None:
                ub_rows = con.execute(
                    "SELECT ts_event, open, high, low, close, volume "
                    "FROM underlying_bars "
                    "WHERE instrument_id = ? AND ts_event >= ? AND ts_event <= ? "
                    "ORDER BY ts_event",
                    [underlying_id, day_start, day_end],
                ).fetchall()
            else:
                ub_rows = []

            if ub_rows:
                candle_json = json.dumps(
                    [
                        {
                            "time": int(r[0].timestamp()),
                            "open": r[1],
                            "high": r[2],
                            "low": r[3],
                            "close": r[4],
                        }
                        for r in ub_rows
                    ]
                )
                volume_json = json.dumps(
                    [
                        {
                            "time": int(r[0].timestamp()),
                            "value": r[5] or 0,
                            "color": "#16A34A50" if r[4] >= r[1] else "#DC262650",
                        }
                        for r in ub_rows
                    ]
                )

            # Breakeven price lines — use actual leg credit sum (not open_mark,
            # which may only reflect one sub-spread for combined positions)
            if legs:
                leg_credit = sum((lg[4] if lg[1][0] == "S" else -lg[4]) for lg in legs)
                credit = abs(leg_credit)
                call_strikes = [lg[6] for lg in legs if lg[5] == "C" and lg[6] is not None]
                put_strikes = [lg[6] for lg in legs if lg[5] == "P" and lg[6] is not None]

                def _sfmt(v: float) -> str:
                    return str(int(v)) if v == int(v) else f"{v:.1f}"

                be_lines = []
                if call_strikes:
                    c_str = min(call_strikes)
                    c_be = round(c_str + credit, 2)
                    be_lines.append(
                        {"price": c_be, "label": f"C-BE {_sfmt(c_str)}+{credit:.1f}={_sfmt(c_be)}"}
                    )
                if put_strikes:
                    p_str = max(put_strikes)
                    p_be = round(p_str - credit, 2)
                    be_lines.append(
                        {"price": p_be, "label": f"P-BE {_sfmt(p_str)}-{credit:.1f}={_sfmt(p_be)}"}
                    )
                be_lines_json = json.dumps(be_lines)

                # Individual strike price reference lines (one per leg)
                sl_items = []
                for lg in legs:
                    rv, sk, act = lg[5], lg[6], lg[1]
                    if sk is None or rv is None:
                        continue
                    pfx = "S" if act and act[0] == "S" else "L"
                    sl_items.append({"price": sk, "label": f"{pfx} {rv}{_sfmt(sk)}"})
                strike_lines_json = json.dumps(sl_items)

                # TP/SL dollar levels for the running P&L pane
                first_qty = legs[0][2]
                first_mult = legs[0][3]
                notional = first_qty * first_mult
                tp_sl = []
                if tp_pct is not None:
                    tp_dollars = leg_credit * float(tp_pct) * notional
                    tp_sl.append({"value": round(tp_dollars, 2), "label": "TP", "color": _GREEN})
                elif tp_abs is not None:
                    tp_dollars = float(tp_abs) * notional
                    tp_sl.append({"value": round(tp_dollars, 2), "label": "TP", "color": _GREEN})
                if sl_pts is not None:
                    sl_dollars = -float(sl_pts) * notional
                    tp_sl.append({"value": round(sl_dollars, 2), "label": "SL", "color": _RED})
                tp_sl_json = json.dumps(tp_sl)

            # Running P&L from option bars (forward-fill prices across legs)
            if legs and open_time and exit_time:
                leg_ids = [lg[0] for lg in legs]
                ph = ", ".join(["?" for _ in leg_ids])
                opt_rows = con.execute(
                    f"SELECT ts_event, instrument_id, close FROM option_bars "
                    f"WHERE instrument_id IN ({ph}) "
                    f"AND ts_event >= ? AND ts_event <= ? "
                    f"ORDER BY ts_event",
                    leg_ids + [open_time, exit_time],
                ).fetchall()

                # inst_id -> (action_prefix, quantity, multiplier, open_price)
                leg_map = {lg[0]: (lg[1][0], lg[2], lg[3], lg[4]) for lg in legs}

                current_prices: dict = {}
                last_ts = int(open_time.timestamp())
                pnl_pts = [{"time": last_ts, "value": 0.0}]

                for ts, grp in _groupby(opt_rows, key=lambda x: x[0]):
                    for _, inst_id, close in grp:
                        if close:  # skip None and zero prices (bad/stale ticks)
                            current_prices[inst_id] = close
                    if len(current_prices) < len(legs):
                        continue
                    ts_int = int(ts.timestamp())
                    if ts_int <= last_ts:
                        continue
                    pnl = 0.0
                    for inst_id, (act_prefix, qty, mult, open_price) in leg_map.items():
                        cur_p = current_prices.get(inst_id, open_price)
                        # S* (STO/STC) = credit sold; B* (BTO/BTC) = debit bought
                        if act_prefix == "S":
                            pnl += (open_price - cur_p) * qty * mult
                        else:
                            pnl += (cur_p - open_price) * qty * mult
                    pnl_pts.append({"time": ts_int, "value": round(pnl, 2)})
                    last_ts = ts_int

                if len(pnl_pts) > 1:
                    pnl_json = json.dumps(pnl_pts)

                    # Hypothetical P&L: what if we hadn't closed at exit_time?
                    # Fetch option bars from exit_time to end-of-session and
                    # continue the same mark calculation using the original open_price.
                    post_rows = con.execute(
                        f"SELECT ts_event, instrument_id, close FROM option_bars "
                        f"WHERE instrument_id IN ({ph}) "
                        f"AND ts_event > ? AND ts_event <= ? "
                        f"ORDER BY ts_event",
                        leg_ids + [exit_time, day_end],
                    ).fetchall()

                    after_pts = [{"time": exit_ts, "value": pnl_pts[-1]["value"]}]
                    for ts, grp in _groupby(post_rows, key=lambda x: x[0]):
                        for _, inst_id, close in grp:
                            if close is not None:
                                current_prices[inst_id] = close
                        ts_int = int(ts.timestamp())
                        pnl = 0.0
                        for inst_id, (act_prefix, qty, mult, open_price) in leg_map.items():
                            cur_p = current_prices.get(inst_id, open_price)
                            if act_prefix == "S":
                                pnl += (open_price - cur_p) * qty * mult
                            else:
                                pnl += (cur_p - open_price) * qty * mult
                        after_pts.append({"time": ts_int, "value": round(pnl, 2)})

                    if len(after_pts) > 1:
                        after_exit_json = json.dumps(after_pts)

        finally:
            con.close()

        reason_colors = {"take_profit": _GREEN, "stop_loss": _RED}
        m_color = reason_colors.get(exit_reason or "", _AMBER)
        markers = [
            {
                "time": int(open_time.timestamp()),
                "position": "belowBar",
                "color": _GREEN,
                "shape": "arrowUp",
                "text": f"Entry {open_mark:.2f}",
            },
            {
                "time": int(exit_time.timestamp()),
                "position": "aboveBar",
                "color": m_color,
                "shape": "arrowDown",
                "text": f"Exit {exit_mark:.2f} – {exit_reason}",
            },
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
        volume_json=volume_json,
        markers_json=markers_json,
        be_lines_json=be_lines_json,
        pnl_json=pnl_json,
        after_exit_json=after_exit_json,
        tp_sl_json=tp_sl_json,
        strike_lines_json=strike_lines_json,
        open_ts=open_ts,
        exit_ts=exit_ts,
    )


# ---------------------------------------------------------------------------
# Comparison helpers (equity curve overlay)
# ---------------------------------------------------------------------------

_PURPLE = "#7C3AED"
_TEAL = "#0891B2"


def _remove_comparison_traces(fig: dict) -> dict:
    """Strip all traces whose name starts with '⊕ ' from a Plotly figure dict."""
    import copy

    fig = copy.deepcopy(fig)
    fig["data"] = [t for t in fig.get("data", []) if not (t.get("name") or "").startswith("⊕ ")]
    return fig


def _fetch_buyhold_trace(
    ticker: str,
    start_date,
    end_date,
    quantity: float = 1.0,
) -> tuple:
    try:
        import yfinance as yf
    except ImportError:
        return None, "yfinance not installed — run: pip install yfinance"

    data = yf.download(
        ticker.upper(), start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    if data.empty:
        return None, f"No data for {ticker!r}"

    close = data["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    close = close.dropna()
    if close.empty:
        return None, f"No valid prices for {ticker!r}"

    first_price = float(close.iloc[0])
    dates = list(close.index)
    pnls = [(float(p) - first_price) * quantity for p in close.values.flatten()]

    qty_label = f" × {quantity:g}" if quantity != 1 else ""
    return (
        go.Scatter(
            x=dates,
            y=pnls,
            mode="lines",
            name=f"⊕ {ticker.upper()} Buy & Hold{qty_label}",
            line=dict(color=_AMBER, width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br><b>$%{y:+,.2f}</b><extra></extra>",
            showlegend=True,
        ),
        None,
    )


def _fetch_livetrades_trace(csv_path: str, start_date, end_date) -> tuple:
    """Load a generic date/pnl CSV and return a cumulative P&L trace.

    Expected CSV format (two columns, header row required):

        date,pnl
        2026-01-15,125.00
        2026-01-22,-48.50

    ``date`` may be ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS``.
    The column named ``date`` (or ``datetime``) and ``pnl`` are matched
    case-insensitively; extra columns are ignored.
    """
    import csv as _csv
    from datetime import datetime as _dt

    path = Path(csv_path).expanduser()
    if not path.exists():
        return None, f"File not found: {csv_path}"

    rows: list[tuple] = []
    try:
        with open(path, newline="") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames:
                return None, "Empty or invalid CSV file"

            fields_lower = {h.strip().lower(): h.strip() for h in reader.fieldnames}
            date_key = next(
                (fields_lower[k] for k in ("date", "datetime", "time", "timestamp") if k in fields_lower),
                None,
            )
            pnl_key = next(
                (fields_lower[k] for k in ("pnl", "net_pnl", "profit", "realized_pnl") if k in fields_lower),
                None,
            )
            if date_key is None:
                return None, "CSV must have a 'date' or 'datetime' column"
            if pnl_key is None:
                return None, "CSV must have a 'pnl' column"

            for row in reader:
                date_str = (row.get(date_key) or "").strip()
                pnl_str = (row.get(pnl_key) or "0").strip()
                try:
                    pnl = float(pnl_str)
                except ValueError:
                    continue
                dt = None
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                    try:
                        dt = _dt.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                if dt is None:
                    continue
                if start_date is not None and dt.date() < start_date:
                    continue
                if end_date is not None and dt.date() > end_date:
                    continue
                rows.append((dt, pnl))
    except Exception as exc:
        return None, f"Error reading CSV: {exc}"

    if not rows:
        return None, "No trades found in the specified date range"

    rows.sort(key=lambda x: x[0])
    total = 0.0
    dates, cumulative = [], []
    for dt, pnl in rows:
        total += pnl
        dates.append(dt)
        cumulative.append(total)

    return (
        go.Scatter(
            x=dates,
            y=cumulative,
            mode="lines+markers",
            name="⊕ Live Trades",
            line=dict(color=_PURPLE, width=1.5),
            marker=dict(size=4),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>$%{y:+,.2f}</b><extra></extra>",
            showlegend=True,
        ),
        None,
    )


def _fetch_backtest_trace(output_db_path: str, compare_bid: int) -> tuple:
    try:
        with OutputDatabase(output_db_path) as odb:
            pp = PostProcessor(odb, backtest_id=compare_bid)
            equity_df = pp.equity_curve()
            initial_equity = pp._load_initial_equity()
            meta_row = odb._con.execute(
                "SELECT strategy_name FROM backtest WHERE id = ?", [compare_bid]
            ).fetchone()
    except Exception as exc:
        return None, f"Error loading backtest #{compare_bid}: {exc}"

    if equity_df.is_empty():
        return None, f"Backtest #{compare_bid} has no trade data"

    times = equity_df["exit_time"].to_list()
    pnls = [float(e) - initial_equity for e in equity_df["equity"].to_list()]
    strategy_name = meta_row[0] if meta_row else "unnamed"

    return (
        go.Scatter(
            x=times,
            y=pnls,
            mode="lines",
            name=f"⊕ #{compare_bid} {strategy_name}",
            line=dict(color=_TEAL, width=1.5, dash="dashdot"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>$%{y:+,.2f}</b><extra></extra>",
            showlegend=True,
        ),
        None,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Study dashboard helpers
# ---------------------------------------------------------------------------


def _extract_swept_params(strategy_params: dict) -> dict:
    """
    Extract sweepable scalar values from a strategy_params model-dump dict,
    keyed by dot-path using the trade/leg names defined in the strategy.
    """
    result = {}
    for trade in strategy_params.get("trades", []):
        tn = trade.get("name", "trade")
        for leg in trade.get("legs", []):
            ln = leg.get("name", "leg")
            for field in ("delta", "dte", "strike_offset"):
                v = leg.get(field)
                if v is not None:
                    result[f"{tn}.{ln}.{field}"] = v
        exit_cfg = trade.get("exit", {}) or {}
        for field in ("stop_loss", "take_profit", "take_profit_pct", "dte_exit"):
            v = exit_cfg.get(field)
            if isinstance(v, dict):  # TakeProfitConfig — extract scalar
                v = v.get("price") or v.get("pct")
            if v is not None:
                result[f"{tn}.exit.{field}"] = v
    return result


def _param_field(dot_key: str) -> str:
    """Convert a dot-path param key to an AgGrid-safe field name (no dots)."""
    return dot_key.replace(".", "_")


def _param_label(dot_key: str) -> str:
    """Human-readable column header: last two path segments (e.g. 'short_put.delta')."""
    parts = dot_key.rsplit(".", 2)
    return ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _build_study_params_df(
    backtests_df,
) -> tuple[list[dict], list[str]]:
    """
    Derive swept parameter columns from per-backtest strategy_params JSON.

    Returns (rows, swept_dot_keys) where each row has backtest_id,
    combination_id, strategy_name, and one key per parameter that varies
    across at least two combinations. Row keys use the original dot-path
    form; callers use _param_field() to get the AgGrid-safe field name.
    """
    import json as _json

    all_rows: list[dict] = []
    for row in backtests_df.iter_rows(named=True):
        params = _extract_swept_params(_json.loads(row["strategy_params"]))
        params["backtest_id"] = row["id"]
        params["combination_id"] = row["combination_id"]
        params["strategy_name"] = row["strategy_name"]
        all_rows.append(params)

    all_param_keys: set[str] = set()
    for row in all_rows:
        all_param_keys.update(
            k for k in row if k not in ("backtest_id", "combination_id", "strategy_name")
        )

    swept: list[str] = []
    for key in sorted(all_param_keys):
        values = {row.get(key) for row in all_rows}
        if len(values) > 1:
            swept.append(key)

    trimmed = [
        {
            "backtest_id": r["backtest_id"],
            "combination_id": r["combination_id"],
            "strategy_name": r["strategy_name"],
            **{k: r.get(k) for k in swept},
        }
        for r in all_rows
    ]
    return trimmed, swept


def _build_study_equity_store(
    all_positions,
    initial_equities: dict,
    param_rows: list[dict],
) -> list[dict]:
    """
    One entry per backtest: ISO timestamps, cumulative P&L series, and a
    short human label built from the swept parameter values.
    """
    import polars as _pl

    label_map: dict[int, str] = {}
    for row in param_rows:
        bid = row["backtest_id"]
        cid = row["combination_id"]
        parts = [f"#{cid}"]
        for k, v in row.items():
            if k in ("backtest_id", "combination_id", "strategy_name") or v is None:
                continue
            short_k = _param_label(k)
            parts.append(f"{short_k}={v:g}" if isinstance(v, float) else f"{short_k}={v}")
        label_map[bid] = " · ".join(parts)

    store: list[dict] = []
    if all_positions.is_empty():
        return store

    for bid in sorted(all_positions["backtest_id"].unique().to_list()):
        sub = all_positions.filter(_pl.col("backtest_id") == bid).sort("exit_time")
        if sub.is_empty():
            continue
        cum_pnl = sub["net_pnl"].cum_sum().to_list()
        times = [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in sub["exit_time"].to_list()]
        store.append({
            "bid": bid,
            "label": label_map.get(bid, f"#{bid}"),
            "times": times,
            "pnls": cum_pnl,
        })
    return store


def _build_study_multi_equity_fig(equity_store: list, selected_bids: set) -> go.Figure:
    """Multi-trace equity (cumulative P&L) figure for the selected backtest IDs."""
    fig = go.Figure()
    visible = [e for e in equity_store if e["bid"] in selected_bids]

    if not visible:
        fig.update_layout(
            annotations=[dict(
                text="Select rows in the table below to display equity curves",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=13, color=_MUTED),
            )],
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=8, r=8, t=8, b=8),
        )
        return fig

    for i, entry in enumerate(visible):
        color = _STUDY_PALETTE[i % len(_STUDY_PALETTE)]
        fig.add_trace(go.Scatter(
            x=entry["times"],
            y=entry["pnls"],
            mode="lines",
            name=entry["label"],
            line=dict(color=color, width=1.5),
            hovertemplate=(
                f"{entry['label']}<br>%{{x|%Y-%m-%d}}"
                "<br><b>$%{y:+,.2f}</b><extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=_GRAY, line_width=1)
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            tickformat="%b %d", tickfont=dict(size=11),
            zeroline=False, title=None,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            tickformat="$+,.0f", tickfont=dict(size=11),
            zeroline=False, title=None,
        ),
        legend=dict(
            font=dict(size=10, family="Inter, system-ui, sans-serif"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=_BORDER,
            borderwidth=1,
        ),
        hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def _fmt_param(v) -> str:
    """Format a swept parameter value for the metrics grid."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _build_study_metrics_grid(
    grid_rows: list[dict],
    swept_cols: list[str],
    multi_strategy: bool,
    initial_selected_bids: set | None = None,
) -> dag.AgGrid:
    """
    AgGrid with one row per combination: checkbox selection (drives equity
    curve), combination link, swept params, and key metrics.

    initial_selected_bids: set of backtest_ids to pre-select on load.
    """
    col_defs = [
        {
            "field": "combo",
            "headerName": "#",
            "width": 75,
            "pinned": "left",
            "cellRenderer": "markdown",
            "checkboxSelection": True,
            "headerCheckboxSelection": True,
            "sortable": False,
            "filter": False,
            "suppressSizeToFit": True,
        },
    ]
    if multi_strategy:
        col_defs.append({
            "field": "strategy_name",
            "headerName": "Strategy",
            "width": 130,
            "cellStyle": {"fontSize": "12px", "color": _MUTED},
        })

    for dot_key in swept_cols:
        col_defs.append({
            "field": _param_field(dot_key),
            "headerName": _param_label(dot_key),
            "width": 90,
            "cellStyle": {
                "textAlign": "right",
                "fontFamily": "monospace",
                "fontSize": "12px",
            },
        })

    col_defs += [
        {
            "field": "net_profit",
            "headerName": "Net P&L",
            "width": 105,
            "cellStyle": {
                "textAlign": "right",
                "fontWeight": "600",
                "fontSize": "12px",
                "styleConditions": [
                    {"condition": "params.value >= 0", "style": {"color": _GREEN}},
                    {"condition": "params.value < 0",  "style": {"color": _RED}},
                ],
            },
            "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
        },
        {
            "field": "win_rate",
            "headerName": "Win %",
            "width": 80,
            "cellStyle": {"textAlign": "right", "fontSize": "12px"},
            "valueFormatter": {"function": "(params.value * 100).toFixed(1) + '%'"},
        },
        {
            "field": "profit_factor",
            "headerName": "PF",
            "width": 72,
            "cellStyle": {"textAlign": "right", "fontSize": "12px"},
            "valueFormatter": {"function": "params.value == null ? '—' : params.value.toFixed(2) + '×'"},
        },
        {
            "field": "sharpe_ratio",
            "headerName": "Sharpe",
            "width": 82,
            "sort": "desc",
            "cellStyle": {
                "textAlign": "right",
                "fontSize": "12px",
                "styleConditions": [
                    {"condition": "params.value >= 1",  "style": {"color": _GREEN}},
                    {"condition": "params.value < 0",   "style": {"color": _RED}},
                ],
            },
            "valueFormatter": {"function": "params.value.toFixed(2)"},
        },
        {
            "field": "max_drawdown_pct",
            "headerName": "Max DD%",
            "width": 90,
            "cellStyle": {"textAlign": "right", "fontSize": "12px", "color": _RED},
            "valueFormatter": {"function": "(params.value * 100).toFixed(1) + '%'"},
        },
        {
            "field": "total_trades",
            "headerName": "Trades",
            "width": 75,
            "cellStyle": {"textAlign": "right", "fontSize": "12px"},
        },
    ]

    rows = []
    for r in grid_rows:
        row: dict = {
            "backtest_id": r["backtest_id"],
            "combo": f"[#{r['combination_id']}](?backtest_id={r['backtest_id']})",
            "strategy_name": r.get("strategy_name", "—"),
            "net_profit": r.get("net_profit", 0.0),
            "win_rate": r.get("percent_profitable", 0.0),
            "profit_factor": r.get("profit_factor"),
            "sharpe_ratio": r.get("sharpe_ratio", 0.0),
            "max_drawdown_pct": r.get("max_drawdown_pct", 0.0),
            "total_trades": r.get("total_trades", 0),
        }
        for dot_key in swept_cols:
            row[_param_field(dot_key)] = _fmt_param(r.get(dot_key))
        rows.append(row)

    pre_selected = (
        [r for r in rows if r["backtest_id"] in initial_selected_bids]
        if initial_selected_bids
        else []
    )

    return dag.AgGrid(
        id="study-metrics-grid",
        rowData=rows,
        columnDefs=col_defs,
        selectedRows=pre_selected,
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        dashGridOptions={
            "rowHeight": 36,
            "headerHeight": 40,
            "animateRows": True,
            "suppressCellFocus": True,
            "domLayout": "autoHeight",
            "rowSelection": "multiple",
        },
        className="ag-theme-alpine",
        style={"width": "100%"},
    )


def _build_study_layout(study_id: int, output_db_path: str) -> html.Div:
    """Study dashboard: multi-curve equity chart + combination metrics table."""
    with OutputDatabase(output_db_path) as odb:
        study_row = odb._con.execute(
            "SELECT name, created_at, total_combinations, finished_at FROM study WHERE id = ?",
            [study_id],
        ).fetchone()
        if study_row is None:
            return html.Div(
                f"Study {study_id} not found.",
                style={"padding": "40px", "fontFamily": "Inter, system-ui, sans-serif"},
            )

        study_name, created_at, total_combinations, finished_at = study_row

        backtests_df = odb._con.execute(
            """
            SELECT id, combination_id, strategy_name, status, strategy_params, initial_equity
            FROM backtest WHERE study_id = ? ORDER BY combination_id
            """,
            [study_id],
        ).pl()

        if backtests_df.is_empty():
            return html.Div(
                f"No combinations found for study {study_id}.",
                style={"padding": "40px", "fontFamily": "Inter, system-ui, sans-serif"},
            )

        pp = PostProcessor(odb, study_id=study_id)
        summary_df = pp.study_summary()

        param_rows, swept_cols = _build_study_params_df(backtests_df)
        param_by_bid = {r["backtest_id"]: r for r in param_rows}

        all_positions = odb._con.execute(
            """
            SELECT p.backtest_id, p.exit_time, p.net_pnl
            FROM position p
            JOIN backtest b ON p.backtest_id = b.id
            WHERE b.study_id = ?
            ORDER BY p.backtest_id, p.exit_time
            """,
            [study_id],
        ).pl()

        initial_equities = {
            row["id"]: float(row["initial_equity"] or 100_000)
            for row in backtests_df.iter_rows(named=True)
        }
        equity_store = _build_study_equity_store(all_positions, initial_equities, param_rows)

    summary_list = summary_df.to_dicts() if not summary_df.is_empty() else []
    grid_rows = []
    for sr in summary_list:
        bid = sr["backtest_id"]
        pr = param_by_bid.get(bid, {})
        merged = {**sr, **{k: pr.get(k) for k in swept_cols}}
        grid_rows.append(merged)

    sorted_rows = sorted(grid_rows, key=lambda r: r.get("sharpe_ratio") or 0.0, reverse=True)
    top5_bids = {r["backtest_id"] for r in sorted_rows[:5]}

    initial_fig = _build_study_multi_equity_fig(equity_store, top5_bids)
    multi_strategy = len({r.get("strategy_name") for r in grid_rows}) > 1
    metrics_grid = _build_study_metrics_grid(grid_rows, swept_cols, multi_strategy, top5_bids)

    created_str = created_at.strftime("%Y-%m-%d %H:%M UTC") if created_at else ""
    n_completed = sum(1 for r in grid_rows if r.get("status") == "completed")

    return html.Div(
        style={
            "fontFamily": "Inter, system-ui, sans-serif",
            "backgroundColor": _BG,
            "minHeight": "100vh",
        },
        children=[
            dcc.Store(id="study-equity-store", data=equity_store),
            # Header
            html.Div(
                className="btkit-header",
                style={
                    "backgroundColor": _HEADER,
                    "color": "white",
                    "padding": "13px 24px",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                },
                children=[
                    html.Span("btkit", style={"fontWeight": "700", "fontSize": "17px", "letterSpacing": "0.04em"}),
                    html.Span("·", style={"color": "#4B5563", "fontSize": "18px"}),
                    html.Span(study_name, style={"fontSize": "14px", "color": "#D1D5DB"}),
                    html.Div(
                        style={"marginLeft": "auto", "display": "flex", "gap": "16px", "alignItems": "center"},
                        children=[
                            html.A(
                                "← All Runs",
                                href="/",
                                style={"fontSize": "12px", "color": "#9CA3AF", "textDecoration": "none"},
                            ),
                            html.Span(
                                f"{n_completed}/{total_combinations} combinations",
                                style={"fontSize": "12px", "color": "#9CA3AF"},
                            ),
                            html.Span(created_str, style={"fontSize": "12px", "color": "#9CA3AF"}),
                        ],
                    ),
                ],
            ),
            # Body
            html.Div(
                style={"padding": "20px", "maxWidth": "1600px", "margin": "0 auto"},
                children=[
                    # Equity curves card
                    _card(
                        [
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "marginBottom": "12px"},
                                children=[
                                    html.P(
                                        "Equity Curves",
                                        style={
                                            "fontSize": "11px",
                                            "fontWeight": "600",
                                            "letterSpacing": "0.08em",
                                            "textTransform": "uppercase",
                                            "color": _MUTED,
                                            "margin": "0",
                                        },
                                    ),
                                    html.Span(
                                        "Top 5 by Sharpe shown · select rows in table to change",
                                        style={"fontSize": "11px", "color": _MUTED, "marginLeft": "12px"},
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="study-equity-fig",
                                figure=initial_fig,
                                config={"displayModeBar": False},
                                style={"height": "320px"},
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),
                    # Combination results card
                    _card(
                        [
                            html.P(
                                "Combination Results",
                                style={
                                    "fontSize": "11px",
                                    "fontWeight": "600",
                                    "letterSpacing": "0.08em",
                                    "textTransform": "uppercase",
                                    "color": _MUTED,
                                    "margin": "0 0 12px 0",
                                },
                            ),
                            metrics_grid,
                        ]
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Backtest index page
# ---------------------------------------------------------------------------


def _build_index_layout(odb) -> html.Div:
    """Landing page: tabbed Backtests / Studies index."""

    # ── Standalone backtests (not part of any study) ─────────────────
    bt_raw = odb._con.execute(
        """
        SELECT
            b.id,
            b.strategy_name,
            b.created_at,
            b.initial_equity,
            COUNT(p.id)                                                                 AS total_trades,
            COALESCE(SUM(p.net_pnl), 0)                                                AS net_profit,
            COUNT(CASE WHEN p.net_pnl > 0 THEN 1 END) * 1.0
                / NULLIF(COUNT(p.id), 0)                                                AS win_rate,
            SUM(CASE WHEN p.net_pnl > 0 THEN p.net_pnl ELSE 0 END)
                / NULLIF(SUM(CASE WHEN p.net_pnl < 0 THEN -p.net_pnl ELSE 0 END), 0)  AS profit_factor,
            SUM(CASE WHEN p.net_pnl < 0 THEN -p.net_pnl ELSE 0 END)
                / NULLIF(COUNT(CASE WHEN p.net_pnl < 0 THEN 1 END), 0)                 AS avg_loss,
            MIN(p.open_time)                                                            AS first_trade,
            MAX(p.exit_time)                                                            AS last_trade
        FROM backtest b
        LEFT JOIN position p ON p.backtest_id = b.id
        WHERE b.study_id IS NULL
        GROUP BY b.id, b.strategy_name, b.created_at, b.initial_equity
        ORDER BY b.created_at DESC, b.id DESC
        """
    ).fetchall()

    bt_rows = []
    for (
        bid, strategy_name, created_at, initial_equity,
        total_trades, net_profit, win_rate, profit_factor, avg_loss,
        first_trade, last_trade,
    ) in bt_raw:
        net_profit = float(net_profit or 0)
        final_equity = float(initial_equity or 0) + net_profit
        if first_trade and last_trade:
            date_range = f"{first_trade.strftime('%Y-%m-%d')} – {last_trade.strftime('%Y-%m-%d')}"
        elif first_trade:
            date_range = f"{first_trade.strftime('%Y-%m-%d')} –"
        else:
            date_range = "—"
        bt_rows.append({
            "id": bid,
            "run": f"[#{bid}](?backtest_id={bid})",
            "strategy": strategy_name or "—",
            "run_date": created_at.strftime("%Y-%m-%d %H:%M") if created_at else "—",
            "date_range": date_range,
            "trades": int(total_trades) if total_trades else 0,
            "net_profit": round(net_profit, 2),
            "win_rate_pct": round(float(win_rate) * 100, 1) if win_rate is not None else 0.0,
            "profit_factor": round(float(profit_factor), 2) if profit_factor is not None else None,
            "avg_loss": round(float(avg_loss), 2) if avg_loss is not None else 0.0,
            "final_equity": round(final_equity, 2),
        })

    # ── Studies ───────────────────────────────────────────────────────
    st_rows: list[dict] = []
    tables = {r[0] for r in odb._con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()}
    if "study" in tables:
        st_raw = odb._con.execute(
            """
            SELECT
                s.id,
                s.name,
                s.created_at,
                s.total_combinations,
                s.finished_at,
                COUNT(DISTINCT CASE WHEN b.status = 'completed' THEN b.id END) AS completed,
                MIN(p.open_time)                                                AS first_trade,
                MAX(p.exit_time)                                                AS last_trade
            FROM study s
            LEFT JOIN backtest b ON b.study_id = s.id
            LEFT JOIN position p ON p.backtest_id = b.id
            GROUP BY s.id, s.name, s.created_at, s.total_combinations, s.finished_at
            ORDER BY s.created_at DESC, s.id DESC
            """
        ).fetchall()
        for (sid, name, created_at, total_combinations, finished_at,
             completed, first_trade, last_trade) in st_raw:
            if first_trade and last_trade:
                date_range = f"{first_trade.strftime('%Y-%m-%d')} – {last_trade.strftime('%Y-%m-%d')}"
            elif first_trade:
                date_range = f"{first_trade.strftime('%Y-%m-%d')} –"
            else:
                date_range = "—"
            st_rows.append({
                "id": sid,
                "study": f"[{name}](?study_id={sid})",
                "run_date": created_at.strftime("%Y-%m-%d %H:%M") if created_at else "—",
                "combinations": f"{int(completed or 0)}/{total_combinations}",
                "date_range": date_range,
                "status": "Complete" if finished_at else "Running",
            })

    # ── Backtest grid ─────────────────────────────────────────────────
    _shared_grid_opts = {
        "rowHeight": 38,
        "headerHeight": 40,
        "animateRows": True,
        "suppressCellFocus": True,
        "domLayout": "autoHeight",
    }
    bt_grid = dag.AgGrid(
        rowData=bt_rows,
        columnDefs=[
            {"field": "run", "headerName": "Run", "width": 70, "cellRenderer": "markdown",
             "sortable": False, "filter": False, "pinned": "left", "suppressSizeToFit": True},
            {"field": "strategy", "headerName": "Strategy", "flex": 1, "minWidth": 120},
            {"field": "run_date", "headerName": "Run Date", "width": 145,
             "cellStyle": {"color": _MUTED, "fontSize": "12px"}},
            {"field": "date_range", "headerName": "Date Range", "width": 220,
             "cellStyle": {"fontFamily": "monospace", "fontSize": "12px"}},
            {"field": "trades", "headerName": "Trades", "width": 80,
             "cellStyle": {"textAlign": "right"}},
            {"field": "net_profit", "headerName": "Net P&L", "width": 115, "sort": "desc",
             "cellStyle": {"textAlign": "right", "fontWeight": "600", "styleConditions": [
                 {"condition": "params.value > 0", "style": {"color": _GREEN}},
                 {"condition": "params.value < 0", "style": {"color": _RED}},
             ]},
             "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"}},
            {"field": "win_rate_pct", "headerName": "Win Rate", "width": 95,
             "cellStyle": {"textAlign": "right"},
             "valueFormatter": {"function": "params.value.toFixed(1) + '%'"}},
            {"field": "profit_factor", "headerName": "Prof. Factor", "width": 105,
             "cellStyle": {"textAlign": "right", "styleConditions": [
                 {"condition": "params.value !== null && params.value >= 1", "style": {"color": _GREEN}},
                 {"condition": "params.value !== null && params.value < 1",  "style": {"color": _RED}},
             ]},
             "valueFormatter": {"function": "params.value !== null ? params.value.toFixed(2) + '×' : '∞'"}},
            {"field": "avg_loss", "headerName": "Avg Loss", "width": 100,
             "cellStyle": {"textAlign": "right", "color": _RED},
             "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"}},
            {"field": "final_equity", "headerName": "Final Equity", "width": 120,
             "cellStyle": {"textAlign": "right"},
             "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"}},
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        dashGridOptions=_shared_grid_opts,
        className="ag-theme-alpine",
        style={"width": "100%"},
    )

    # ── Studies grid ──────────────────────────────────────────────────
    st_grid = dag.AgGrid(
        rowData=st_rows,
        columnDefs=[
            {"field": "study", "headerName": "Study", "flex": 1, "minWidth": 160,
             "cellRenderer": "markdown", "sortable": False, "filter": False},
            {"field": "run_date", "headerName": "Run Date", "width": 145,
             "cellStyle": {"color": _MUTED, "fontSize": "12px"}},
            {"field": "combinations", "headerName": "Combinations", "width": 130,
             "cellStyle": {"textAlign": "right", "fontFamily": "monospace"}},
            {"field": "date_range", "headerName": "Date Range", "width": 220,
             "cellStyle": {"fontFamily": "monospace", "fontSize": "12px"}},
            {"field": "status", "headerName": "Status", "width": 100,
             "cellStyle": {"styleConditions": [
                 {"condition": "params.value === 'Complete'", "style": {"color": _GREEN, "fontWeight": "600"}},
                 {"condition": "params.value !== 'Complete'", "style": {"color": _AMBER}},
             ]}},
        ],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        dashGridOptions=_shared_grid_opts,
        className="ag-theme-alpine",
        style={"width": "100%"},
    )

    n_bt = len(bt_rows)
    n_st = len(st_rows)
    default_tab = "studies" if st_rows else "backtests"

    _tab_style = {"fontSize": "13px", "padding": "8px 16px"}
    _tab_selected = {**_tab_style, "fontWeight": "600", "borderTop": f"2px solid {_BLUE}"}

    return html.Div(
        style={"fontFamily": "Inter, system-ui, sans-serif", "backgroundColor": _BG, "minHeight": "100vh"},
        children=[
            html.Div(
                className="btkit-header",
                style={"backgroundColor": _HEADER, "color": "white", "padding": "13px 24px",
                       "display": "flex", "alignItems": "center", "gap": "12px"},
                children=[
                    html.Span("btkit", style={"fontWeight": "700", "fontSize": "17px", "letterSpacing": "0.04em"}),
                    html.Span("·", style={"color": "#4B5563", "fontSize": "18px"}),
                    html.Span("Runs", style={"fontSize": "14px", "color": "#D1D5DB"}),
                    html.Span(
                        f"{n_bt} backtest{'s' if n_bt != 1 else ''}  ·  {n_st} stud{'ies' if n_st != 1 else 'y'}",
                        style={"marginLeft": "auto", "fontSize": "12px", "color": "#9CA3AF"},
                    ),
                ],
            ),
            html.Div(
                style={"padding": "24px", "maxWidth": "1400px", "margin": "0 auto"},
                children=[
                    dcc.Tabs(
                        value=default_tab,
                        colors={"border": _BORDER, "primary": _BLUE, "background": _CARD},
                        style={"marginBottom": "0"},
                        children=[
                            dcc.Tab(
                                label=f"Backtests ({n_bt})",
                                value="backtests",
                                style=_tab_style,
                                selected_style=_tab_selected,
                                children=[_card(
                                    [
                                        html.P(
                                            "Click a run number to open its full dashboard",
                                            style={"fontSize": "11px", "fontWeight": "600",
                                                   "letterSpacing": "0.08em", "textTransform": "uppercase",
                                                   "color": _MUTED, "margin": "0 0 14px 0"},
                                        ),
                                        bt_grid,
                                    ],
                                    style={"borderTopLeftRadius": "0", "borderTopRightRadius": "0"},
                                )],
                            ),
                            dcc.Tab(
                                label=f"Studies ({n_st})",
                                value="studies",
                                style=_tab_style,
                                selected_style=_tab_selected,
                                children=[_card(
                                    [
                                        html.P(
                                            "Click a study name to open its dashboard",
                                            style={"fontSize": "11px", "fontWeight": "600",
                                                   "letterSpacing": "0.08em", "textTransform": "uppercase",
                                                   "color": _MUTED, "margin": "0 0 14px 0"},
                                        ),
                                        st_grid,
                                    ],
                                    style={"borderTopLeftRadius": "0", "borderTopRightRadius": "0"},
                                )],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_backtest_layout(
    effective_bid: int,
    output_db_path: str,
    input_db_path: str | None,
) -> html.Div:
    """Build and return the full Dash layout for a specific backtest run."""
    with OutputDatabase(output_db_path) as odb:
        pp = PostProcessor(odb, backtest_id=effective_bid)
        bid = pp._resolve_backtest_id()
        m = pp.metrics()
        equity_df = pp.equity_curve()
        initial_equity = pp._load_initial_equity()
        net_pnls = pp._load_positions()["net_pnl"].to_list()

        positions = odb._con.execute(
            "SELECT id, trade_name, open_time, exit_time, exit_reason, "
            "open_mark, exit_mark, net_pnl, worst_mark "
            "FROM position WHERE backtest_id = ? ORDER BY open_time",
            [bid],
        ).pl()

        leg_rows = odb._con.execute(
            'SELECT pl.position_id, pl."right", pl.strike_price, pl.expiration, pl.action '
            "FROM position_leg pl "
            "JOIN position p ON pl.position_id = p.id "
            "WHERE p.backtest_id = ? "
            "ORDER BY pl.position_id, pl.strike_price",
            [bid],
        ).fetchall()

        from collections import defaultdict as _dd

        leg_strings: dict[int, str] = {}
        grouped: dict[int, list] = _dd(list)
        for pid, right, strike, expiry, action in leg_rows:
            grouped[pid].append((right, strike, expiry, action))
        for pid, legs_list in grouped.items():
            parts = []
            for right, strike, expiry, action in legs_list:
                prefix = "+" if action and action[0] == "B" else "−"
                exp_str = expiry.strftime("%-m/%-d") if expiry else ""
                strike_str = f"{int(strike)}" if strike == int(strike) else f"{strike:.1f}"
                parts.append(f"{prefix}{right}{strike_str} {exp_str}")
            leg_strings[pid] = "  ".join(parts)

        meta_row = odb._con.execute(
            "SELECT strategy_name, created_at, strategy_params FROM backtest WHERE id = ?", [bid]
        ).fetchone()
        strategy_name = meta_row[0] if meta_row else "—"
        created_at = (
            meta_row[1].strftime("%Y-%m-%d %H:%M UTC") if (meta_row and meta_row[1]) else ""
        )
        _raw_params = meta_row[2] if meta_row else None

        pos_leg_map = odb._con.execute(
            "SELECT DISTINCT pl.position_id, pl.instrument_id FROM position_leg pl "
            "JOIN position p ON pl.position_id = p.id "
            "WHERE p.backtest_id = ?",
            [bid],
        ).fetchall()

        other_bt_rows = odb._con.execute(
            "SELECT id, strategy_name, created_at FROM backtest "
            "WHERE id != ? ORDER BY created_at DESC",
            [bid],
        ).fetchall()

    compare_bt_options = [
        {
            "label": f"#{r[0]}  ·  {r[1] or 'unnamed'}  ·  {r[2].strftime('%Y-%m-%d') if r[2] else '—'}",
            "value": r[0],
        }
        for r in other_bt_rows
    ]

    # ── Underlying symbol lookup ───────────────────────────────────────
    underlying_symbols: dict[int, str] = {}
    if input_db_path and pos_leg_map:
        import duckdb as _ddb

        inst_ids = list({r[1] for r in pos_leg_map})
        _icon = _ddb.connect(input_db_path, read_only=True)
        try:
            # LIMIT 1 per option avoids a full-table scan of the 4M+ row
            # option_bars table. We only need one row to know underlying_id,
            # then resolve the symbol from the much smaller underlying_bars.
            underlying_ids: dict[int, int] = {}
            for iid in inst_ids:
                row = _icon.execute(
                    "SELECT underlying_id FROM option_bars WHERE instrument_id = ? LIMIT 1",
                    [iid],
                ).fetchone()
                if row:
                    underlying_ids[iid] = row[0]

            iid_to_sym: dict[int, str] = {}
            if underlying_ids:
                uid_list = list(set(underlying_ids.values()))
                ph = ",".join("?" * len(uid_list))
                sym_rows = _icon.execute(
                    f"SELECT DISTINCT instrument_id, symbol FROM underlying_bars"
                    f" WHERE instrument_id IN ({ph})",
                    uid_list,
                ).fetchall()
                uid_to_sym = dict(sym_rows)
                iid_to_sym = {
                    iid: uid_to_sym[uid]
                    for iid, uid in underlying_ids.items()
                    if uid in uid_to_sym
                }
        finally:
            _icon.close()
        for pid, iid in pos_leg_map:
            if iid in iid_to_sym and pid not in underlying_symbols:
                underlying_symbols[pid] = iid_to_sym[iid]

    # ── Build figures ──────────────────────────────────────────────────
    equity_fig = _build_equity_chart(equity_df, initial_equity)
    boot_fig = _build_bootstrap_fig(net_pnls, initial_equity)
    metrics_tbl = _build_metrics_table(m)

    import json as _json

    _params_dict = _json.loads(_raw_params) if _raw_params else {}
    params_panel = _build_params_panel(_params_dict)
    trade_tbl = _build_trade_table(positions, leg_strings, underlying_symbols)

    _open_times = positions["open_time"].drop_nulls()
    _exit_times = positions["exit_time"].drop_nulls()
    _meta_store = {
        "start_date": _open_times.min().date().isoformat() if not _open_times.is_empty() else None,
        "end_date": _exit_times.max().date().isoformat() if not _exit_times.is_empty() else None,
        "initial_equity": initial_equity,
    }

    nav_hint = html.A(
        "← All Runs",
        href="/",
        style={"fontSize": "12px", "color": "#9CA3AF", "textDecoration": "none"},
    )

    # ── Layout ────────────────────────────────────────────────────────
    return html.Div(
        style={
            "fontFamily": "Inter, system-ui, sans-serif",
            "backgroundColor": _BG,
            "minHeight": "100vh",
        },
        children=[
            dcc.Store(id="btkit-meta", data=_meta_store),
            dcc.Store(id="btkit-net-pnls", data=net_pnls),
            # Header
            html.Div(
                className="btkit-header",
                style={
                    "backgroundColor": _HEADER,
                    "color": "white",
                    "padding": "13px 24px",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                },
                children=[
                    html.Span(
                        "btkit",
                        style={"fontWeight": "700", "fontSize": "17px", "letterSpacing": "0.04em"},
                    ),
                    html.Span("·", style={"color": "#4B5563", "fontSize": "18px"}),
                    html.Span(strategy_name, style={"fontSize": "14px", "color": "#D1D5DB"}),
                    html.Div(
                        className="btkit-header-meta",
                        style={"marginLeft": "auto", "display": "flex", "gap": "16px", "alignItems": "center"},
                        children=[
                            nav_hint,
                            html.Span(
                                f"run #{bid}", style={"fontSize": "12px", "color": "#9CA3AF"}
                            ),
                            html.Span(created_at, style={"fontSize": "12px", "color": "#9CA3AF"}),
                        ],
                    ),
                ],
            ),
            # Body
            html.Div(
                className="btkit-body",
                style={"padding": "20px", "maxWidth": "1600px", "margin": "0 auto"},
                children=[
                    # Row 1: Equity curve + Bootstrap fan
                    html.Div(
                        className="btkit-row",
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                        children=[
                            _card(
                                [
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginBottom": "12px",
                                        },
                                        children=[
                                            html.P(
                                                "Equity Curve",
                                                style={
                                                    "fontSize": "11px",
                                                    "fontWeight": "600",
                                                    "letterSpacing": "0.08em",
                                                    "textTransform": "uppercase",
                                                    "color": _MUTED,
                                                    "margin": "0",
                                                },
                                            ),
                                            html.Button(
                                                "Compare",
                                                id="compare-btn",
                                                n_clicks=0,
                                                style={
                                                    "marginLeft": "auto",
                                                    "fontSize": "11px",
                                                    "fontWeight": "600",
                                                    "letterSpacing": "0.04em",
                                                    "color": _BLUE,
                                                    "background": "none",
                                                    "border": f"1px solid {_BLUE}",
                                                    "borderRadius": "4px",
                                                    "padding": "3px 10px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(
                                        id="equity-fig",
                                        figure=equity_fig,
                                        config={"displayModeBar": False},
                                        className="btkit-chart-equity",
                                        style={"height": "280px"},
                                    ),
                                    html.Div(
                                        id="compare-panel",
                                        style={
                                            "display": "none",
                                            "marginTop": "12px",
                                            "padding": "12px 14px",
                                            "backgroundColor": _BG,
                                            "borderRadius": "6px",
                                            "border": f"1px solid {_BORDER}",
                                        },
                                        children=[
                                            dcc.RadioItems(
                                                id="compare-type",
                                                options=[
                                                    {"label": " Buy & Hold", "value": "buyhold"},
                                                    {"label": " Live Trades (CSV)", "value": "livetrades"},
                                                    {"label": " Backtest", "value": "backtest"},
                                                ],
                                                value="buyhold",
                                                inline=True,
                                                style={
                                                    "fontSize": "12px",
                                                    "color": _TEXT,
                                                    "marginBottom": "10px",
                                                },
                                                inputStyle={"marginRight": "4px", "marginLeft": "12px"},
                                            ),
                                            html.Div(
                                                id="compare-ticker-div",
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "gap": "8px",
                                                    "marginBottom": "10px",
                                                },
                                                children=[
                                                    html.Label(
                                                        "Ticker",
                                                        style={
                                                            "fontSize": "12px",
                                                            "color": _MUTED,
                                                            "width": "60px",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="compare-ticker",
                                                        type="text",
                                                        placeholder="e.g. SPY",
                                                        value="SPY",
                                                        debounce=False,
                                                        style={
                                                            "width": "90px",
                                                            "fontSize": "12px",
                                                            "padding": "4px 8px",
                                                            "border": f"1px solid {_BORDER}",
                                                            "borderRadius": "4px",
                                                        },
                                                    ),
                                                    html.Label(
                                                        "Qty",
                                                        style={
                                                            "fontSize": "12px",
                                                            "color": _MUTED,
                                                            "marginLeft": "10px",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="compare-qty",
                                                        type="number",
                                                        value=1,
                                                        min=0.01,
                                                        step=1,
                                                        debounce=False,
                                                        style={
                                                            "width": "80px",
                                                            "fontSize": "12px",
                                                            "padding": "4px 8px",
                                                            "border": f"1px solid {_BORDER}",
                                                            "borderRadius": "4px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="compare-csv-div",
                                                style={
                                                    "display": "none",
                                                    "alignItems": "center",
                                                    "gap": "8px",
                                                    "marginBottom": "10px",
                                                },
                                                children=[
                                                    html.Label(
                                                        "CSV Path",
                                                        style={
                                                            "fontSize": "12px",
                                                            "color": _MUTED,
                                                            "width": "60px",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="compare-csv",
                                                        type="text",
                                                        placeholder="~/trades.csv",
                                                        debounce=False,
                                                        style={
                                                            "width": "340px",
                                                            "fontSize": "12px",
                                                            "padding": "4px 8px",
                                                            "border": f"1px solid {_BORDER}",
                                                            "borderRadius": "4px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="compare-backtest-div",
                                                style={
                                                    "display": "none",
                                                    "alignItems": "center",
                                                    "gap": "8px",
                                                    "marginBottom": "10px",
                                                },
                                                children=[
                                                    html.Label(
                                                        "Backtest",
                                                        style={
                                                            "fontSize": "12px",
                                                            "color": _MUTED,
                                                            "width": "60px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="compare-backtest-id",
                                                        options=compare_bt_options,
                                                        value=compare_bt_options[0]["value"] if compare_bt_options else None,
                                                        placeholder="Select a backtest…",
                                                        clearable=False,
                                                        style={
                                                            "width": "420px",
                                                            "fontSize": "12px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "gap": "10px",
                                                },
                                                children=[
                                                    html.Button(
                                                        "Load",
                                                        id="compare-load-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "fontSize": "11px",
                                                            "fontWeight": "600",
                                                            "color": "white",
                                                            "background": _BLUE,
                                                            "border": "none",
                                                            "borderRadius": "4px",
                                                            "padding": "5px 16px",
                                                            "cursor": "pointer",
                                                        },
                                                    ),
                                                    html.Button(
                                                        "Clear",
                                                        id="compare-clear-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "fontSize": "11px",
                                                            "fontWeight": "600",
                                                            "color": _MUTED,
                                                            "background": "none",
                                                            "border": f"1px solid {_BORDER}",
                                                            "borderRadius": "4px",
                                                            "padding": "5px 16px",
                                                            "cursor": "pointer",
                                                        },
                                                    ),
                                                    html.Span(
                                                        id="compare-status",
                                                        style={"fontSize": "12px", "color": _RED},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            _card(
                                [
                                    _section_label(
                                        f"Bootstrap Equity Fan  ·  1 000 resamples  ·  "
                                        f"{len(net_pnls)} trades"
                                    ),
                                    dcc.Graph(
                                        figure=boot_fig,
                                        config={"displayModeBar": False},
                                        className="btkit-chart-bootstrap",
                                        style={"height": "280px"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                    ),
                    # Row 2: [Params | Metrics] (flex:1) | P&L Distribution (flex:1)
                    # The left wrapper is flex:1, matching the equity card in Row 1 exactly.
                    html.Div(
                        className="btkit-row",
                        style={"display": "flex", "gap": "16px", "alignItems": "stretch", "marginBottom": "16px"},
                        children=[
                            # Left half: Params + Metrics side-by-side
                            html.Div(
                                style={"flex": "1", "minWidth": "0", "display": "flex", "gap": "16px"},
                                children=[
                                    _card(
                                        html.Div([
                                            params_panel,
                                        ], style={"overflowY": "auto", "maxHeight": "320px"}),
                                        style={"flex": "1", "minWidth": "0"},
                                    ),
                                    _card(
                                        html.Div([
                                            metrics_tbl,
                                        ], style={"overflowY": "auto", "maxHeight": "320px"}),
                                        style={"flex": "1", "minWidth": "0"},
                                    ),
                                ],
                            ),
                            # Right half: P&L Distribution
                            _card(
                                [
                                    html.Div(
                                        className="btkit-hist-controls",
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginBottom": "12px",
                                        },
                                        children=[
                                            _section_label("P&L Distribution"),
                                            dcc.RadioItems(
                                                id="hist-mode",
                                                options=[
                                                    {"label": " All Trades", "value": "all"},
                                                    {"label": " Winners", "value": "winners"},
                                                    {"label": " Losers", "value": "losers"},
                                                ],
                                                value="all",
                                                inline=True,
                                                style={
                                                    "marginLeft": "auto",
                                                    "fontSize": "12px",
                                                    "color": _MUTED,
                                                    "gap": "14px",
                                                },
                                                inputStyle={"marginRight": "4px"},
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(
                                        id="pnl-hist",
                                        config={"displayModeBar": False},
                                        className="btkit-chart-histogram",
                                        style={"height": "220px"},
                                    ),
                                ],
                                style={"flex": "1", "minWidth": "0"},
                            ),
                        ],
                    ),
                    # Row 3: All Trades (full width)
                    _card(
                        [
                            _section_label(
                                f"All Trades  ·  {m['total_trades']} positions"
                                + (
                                    "  ·  📈 click row chart icon to open trade chart"
                                    if input_db_path
                                    else ""
                                )
                            ),
                            trade_tbl,
                        ],
                        className="btkit-trade-grid",
                    ),
                ],
            ),
        ],
    )


def create_app(
    output_db_path: str,
    input_db_path: str | None = None,
    backtest_id: int | None = None,
) -> Dash:
    """Build and return the Dash app.

    ``app.layout`` returns a thin shell with ``dcc.Location`` and a
    page-content div. A routing callback reads ``?backtest_id=N`` from the
    URL and renders either the landing-page index or the full backtest
    dashboard — the canonical Dash pattern for client-side URL routing.
    """
    app = Dash(
        __name__,
        assets_folder=_ASSETS,
        title="btkit",
        suppress_callback_exceptions=True,
    )
    server = app.server

    @server.route("/chart/<int:position_id>")
    def trade_chart_page(position_id: int):
        html_content = _build_chart_html(position_id, output_db_path, input_db_path)
        return Response(html_content, content_type="text/html; charset=utf-8")

    def _serve_layout():
        return html.Div(
            [
                dcc.Location(id="url", refresh=False),
                dcc.Loading(
                    html.Div(id="page-content"),
                    type="circle",
                    color=_BLUE,
                    style={"minHeight": "100vh"},
                ),
            ],
            style={"minHeight": "100vh", "backgroundColor": _BG},
        )

    app.layout = _serve_layout

    @app.callback(
        Output("page-content", "children"),
        Input("url", "search"),
    )
    def _route(search: str | None):
        from urllib.parse import parse_qs

        params = parse_qs((search or "").lstrip("?"))
        bid_str = params.get("backtest_id", [None])[0]
        sid_str = params.get("study_id", [None])[0]
        effective_bid = int(bid_str) if bid_str else backtest_id
        if sid_str:
            return _build_study_layout(int(sid_str), output_db_path)
        if effective_bid is None:
            with OutputDatabase(output_db_path) as odb:
                return _build_index_layout(odb)
        return _build_backtest_layout(effective_bid, output_db_path, input_db_path)

    # ── Study equity callback ────────────────────────────────────────────
    @app.callback(
        Output("study-equity-fig", "figure"),
        Input("study-metrics-grid", "selectedRows"),
        State("study-equity-store", "data"),
        prevent_initial_call=True,
    )
    def _update_study_equity(selected_rows, equity_store):
        if not equity_store:
            return go.Figure()
        selected_bids = {row["backtest_id"] for row in (selected_rows or [])}
        if not selected_bids:
            return go.Figure()
        return _build_study_multi_equity_fig(equity_store, selected_bids)

    # ── Histogram callback ───────────────────────────────────────────────
    @app.callback(
        Output("pnl-hist", "figure"),
        Input("hist-mode", "value"),
        State("btkit-net-pnls", "data"),
    )
    def update_histogram(mode: str, net_pnls_data: list) -> go.Figure:
        return _build_pnl_histogram(net_pnls_data or [], mode)

    # ── Compare panel toggle ─────────────────────────────────────────────
    @app.callback(
        Output("compare-panel", "style"),
        Input("compare-btn", "n_clicks"),
        State("compare-panel", "style"),
        prevent_initial_call=True,
    )
    def _toggle_compare_panel(_n, current_style):
        visible = current_style.get("display") != "none"
        return {**current_style, "display": "none" if visible else "block"}

    # ── Compare input type toggle ────────────────────────────────────────
    @app.callback(
        Output("compare-ticker-div", "style"),
        Output("compare-csv-div", "style"),
        Output("compare-backtest-div", "style"),
        Input("compare-type", "value"),
    )
    def _toggle_compare_inputs(mode):
        base = {"alignItems": "center", "gap": "8px", "marginBottom": "10px"}
        return (
            {**base, "display": "flex" if mode == "buyhold" else "none"},
            {**base, "display": "flex" if mode == "livetrades" else "none"},
            {**base, "display": "flex" if mode == "backtest" else "none"},
        )

    # ── Load / clear comparison trace ───────────────────────────────────
    @app.callback(
        Output("equity-fig", "figure"),
        Output("compare-status", "children"),
        Input("compare-load-btn", "n_clicks"),
        Input("compare-clear-btn", "n_clicks"),
        State("equity-fig", "figure"),
        State("compare-type", "value"),
        State("compare-ticker", "value"),
        State("compare-qty", "value"),
        State("compare-csv", "value"),
        State("compare-backtest-id", "value"),
        State("btkit-meta", "data"),
        prevent_initial_call=True,
    )
    def _load_comparison(_lc, _cc, current_fig, mode, ticker, qty, csv_path, compare_bid, meta):
        fig = _remove_comparison_traces(current_fig)
        if ctx.triggered_id == "compare-clear-btn":
            return fig, ""

        start_date = end_date = None
        if meta:
            from datetime import date as _date

            try:
                start_date = _date.fromisoformat(meta["start_date"]) if meta.get("start_date") else None
                end_date = _date.fromisoformat(meta["end_date"]) if meta.get("end_date") else None
            except Exception:
                pass

        if mode == "buyhold":
            if not ticker:
                return fig, "Enter a ticker symbol"
            quantity = float(qty) if qty is not None else 1.0
            trace, err = _fetch_buyhold_trace(ticker, start_date, end_date, quantity)
        elif mode == "livetrades":
            if not csv_path:
                return fig, "Enter a CSV file path"
            trace, err = _fetch_livetrades_trace(csv_path, start_date, end_date)
        else:
            if compare_bid is None:
                return fig, "Select a backtest to compare"
            trace, err = _fetch_backtest_trace(output_db_path, compare_bid)

        if err:
            return fig, err

        fig["data"].append(trace.to_plotly_json())
        return fig, ""

    return app


def run_dashboard(
    output_db_path: str,
    input_db_path: str | None = None,
    backtest_id: int | None = None,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Build the Dash app and start the development server."""
    app = create_app(output_db_path, input_db_path=input_db_path, backtest_id=backtest_id)
    print(f"  Dashboard: http://localhost:{port}")
    app.run(port=port, debug=debug)
