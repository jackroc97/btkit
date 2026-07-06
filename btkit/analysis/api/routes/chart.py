"""Per-trade chart data endpoint (requires BTKIT_INPUT_DB env var)."""

from __future__ import annotations

import math
from datetime import UTC
from itertools import groupby

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..db import query as out_query
from ..db_input import connect as input_connect
from .detail import _extract_params

router = APIRouter()

UTC = UTC

_GREEN = "#4ade80"
_RED = "#f87171"
_AMBER = "#94a3b8"

_POS_SQL = """
SELECT p.open_time, p.exit_time, p.exit_reason, p.open_mark, p.exit_mark, p.net_pnl,
       b.strategy_params
FROM position p
JOIN backtest b ON b.id = p.backtest_id
WHERE p.id = ?
"""

_LEGS_SQL = """
SELECT instrument_id, action, quantity, multiplier, open_price, "right", strike_price
FROM position_leg
WHERE position_id = ?
ORDER BY id
"""

_UB_SQL = """
SELECT ts_event, open, high, low, close, volume
FROM underlying_bars
WHERE instrument_id = ? AND ts_event >= ? AND ts_event <= ?
ORDER BY ts_event
"""

_OB_SQL = """
SELECT ts_event, instrument_id, close
FROM option_bars
WHERE instrument_id IN ({ph}) AND ts_event >= ? AND ts_event <= ?
ORDER BY ts_event
"""

_OPT_BARS_SQL = """
SELECT ts_event, open, high, low, close, volume
FROM option_bars
WHERE instrument_id = ? AND ts_event >= ? AND ts_event <= ?
ORDER BY ts_event
"""


def _sfmt(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:.1f}"


def _nan_to_none(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


@router.get("/positions/{position_id}/chart")
def position_chart(
    position_id: int,
    leg_id: int | None = Query(None),
) -> JSONResponse:
    con = input_connect()
    if con is None:
        return JSONResponse({"has_data": False, "reason": "no_input_db"})

    pos_cols, pos_rows = out_query(_POS_SQL, [position_id])
    if not pos_rows:
        con.close()
        raise HTTPException(status_code=404, detail="Position not found")

    pos = dict(zip(pos_cols, pos_rows[0], strict=False))
    leg_cols, leg_rows = out_query(_LEGS_SQL, [position_id])
    legs = [dict(zip(leg_cols, r, strict=False)) for r in leg_rows]

    try:
        return _build(pos, legs, con, leg_id=leg_id)
    except Exception as exc:
        return JSONResponse({"has_data": False, "reason": "error", "detail": str(exc)})
    finally:
        con.close()


def _build(pos: dict, legs: list[dict], con, leg_id: int | None = None) -> JSONResponse:
    open_time = pos["open_time"]
    exit_time = pos["exit_time"]
    exit_reason = pos.get("exit_reason") or ""
    open_mark = _nan_to_none(pos.get("open_mark")) or 0.0
    exit_mark = _nan_to_none(pos.get("exit_mark")) or 0.0

    if not open_time or not exit_time:
        return JSONResponse({"has_data": False, "reason": "missing_timestamps"})

    open_ts = int(open_time.timestamp())
    exit_ts = int(exit_time.timestamp())

    day_start = open_time.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
    day_end = exit_time.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=UTC)

    params = _extract_params(pos.get("strategy_params"))

    # In leg mode only the selected leg is used for P&L; spread mode uses all legs.
    active_legs = [lg for lg in legs if lg["instrument_id"] == leg_id] if leg_id else legs

    # ── Underlying bars ───────────────────────────────────────────────────────
    candles: list[dict] = []
    volume: list[dict] = []
    underlying_id = None

    if legs:
        uid_row = con.execute(
            "SELECT underlying_id FROM option_bars WHERE instrument_id = ? LIMIT 1",
            [legs[0]["instrument_id"]],
        ).fetchone()
        if uid_row:
            underlying_id = uid_row[0]

    if underlying_id is not None:
        for r in con.execute(_UB_SQL, [underlying_id, day_start, day_end]).fetchall():
            ts = int(r[0].timestamp())
            candles.append({"time": ts, "open": r[1], "high": r[2], "low": r[3], "close": r[4]})
            color = "rgba(22,163,74,0.35)" if r[4] >= r[1] else "rgba(220,38,38,0.35)"
            volume.append({"time": ts, "value": r[5] or 0, "color": color})

    if not candles:
        return JSONResponse({"has_data": False, "reason": "no_underlying_bars"})

    # ── Breakeven lines (spread mode only) ────────────────────────────────────
    be_lines: list[dict] = []
    if legs and not leg_id:
        leg_credit = sum(
            (lg["open_price"] if lg["action"][0] == "S" else -lg["open_price"])
            for lg in legs
            if lg.get("open_price")
        )
        credit = abs(leg_credit)
        call_strikes = [
            lg["strike_price"] for lg in legs if lg.get("right") == "C" and lg.get("strike_price")
        ]
        put_strikes = [
            lg["strike_price"] for lg in legs if lg.get("right") == "P" and lg.get("strike_price")
        ]

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

    # ── Strike lines (active legs only) ───────────────────────────────────────
    strike_lines: list[dict] = []
    for lg in active_legs:
        sk = lg.get("strike_price")
        rv = lg.get("right")
        if sk is None or rv is None:
            continue
        pfx = "S" if lg["action"][0] == "S" else "L"
        strike_lines.append({"price": sk, "label": f"{pfx} {rv}{_sfmt(sk)}"})

    # ── TP / SL dollar lines (spread mode only) ───────────────────────────────
    tp_sl_lines: list[dict] = []
    if legs and not leg_id:
        first_qty = legs[0].get("quantity") or 1
        first_mult = legs[0].get("multiplier") or 100
        notional = first_qty * first_mult
        leg_credit_raw = sum(
            (lg["open_price"] if lg["action"][0] == "S" else -lg["open_price"])
            for lg in legs
            if lg.get("open_price")
        )
        tp_pct = params.get("take_profit_pct")
        if tp_pct:
            tp_sl_lines.append(
                {
                    "value": round(leg_credit_raw * float(tp_pct) * notional, 2),
                    "label": "TP",
                    "color": _GREEN,
                }
            )
        sl_pts = params.get("stop_loss")
        if sl_pts:
            tp_sl_lines.append(
                {
                    "value": round(-float(sl_pts) * notional, 2),
                    "label": "SL",
                    "color": _RED,
                }
            )

    # ── Leg mode: replace underlying candles with option OHLCV ───────────────
    if leg_id and active_legs:
        ob_rows = con.execute(_OPT_BARS_SQL, [leg_id, day_start, day_end]).fetchall()
        opt_candles = []
        opt_volume = []
        for r in ob_rows:
            o, h, low, c, vol = r[1], r[2], r[3], r[4], r[5]
            if o is None or c is None:
                continue
            ts = int(r[0].timestamp())
            opt_candles.append({"time": ts, "open": o, "high": h or c, "low": low or c, "close": c})
            color = "rgba(22,163,74,0.35)" if c >= o else "rgba(220,38,38,0.35)"
            opt_volume.append({"time": ts, "value": vol or 0, "color": color})
        if opt_candles:
            candles = opt_candles
            volume = opt_volume
            strike_lines = []  # strike levels are underlying-referenced, irrelevant here

    # ── Running P&L from option bars (active legs) ───────────────────────────
    pnl_pts: list[dict] = []
    after_exit_pts: list[dict] = []

    if active_legs:
        leg_ids = [lg["instrument_id"] for lg in active_legs]
        leg_map = {
            lg["instrument_id"]: (
                lg["action"][0],
                lg.get("quantity") or 1,
                lg.get("multiplier") or 100,
                lg.get("open_price") or 0.0,
            )
            for lg in active_legs
        }
        ph = ", ".join("?" for _ in leg_ids)

        opt_rows = con.execute(
            _OB_SQL.format(ph=ph),
            leg_ids + [open_time, exit_time],
        ).fetchall()

        current_prices: dict[int, float] = {}
        last_ts = open_ts
        pnl_pts = [{"time": open_ts, "value": 0.0}]

        for ts_val, grp in groupby(opt_rows, key=lambda x: x[0]):
            for _, inst_id, close in grp:
                if close:
                    current_prices[inst_id] = close
            if len(current_prices) < len(active_legs):
                continue
            ts_int = int(ts_val.timestamp())
            if ts_int <= last_ts:
                continue
            pnl = 0.0
            for inst_id, (act, qty, mult, op) in leg_map.items():
                cur = current_prices.get(inst_id, op)
                pnl += (op - cur) * qty * mult if act == "S" else (cur - op) * qty * mult
            pnl_pts.append({"time": ts_int, "value": round(pnl, 2)})
            last_ts = ts_int

        if len(pnl_pts) > 1:
            post_rows = con.execute(
                _OB_SQL.format(ph=ph),
                leg_ids + [exit_time, day_end],
            ).fetchall()

            after_exit_pts = [{"time": exit_ts, "value": pnl_pts[-1]["value"]}]
            for ts_val, grp in groupby(post_rows, key=lambda x: x[0]):
                for _, inst_id, close in grp:
                    if close is not None:
                        current_prices[inst_id] = close
                ts_int = int(ts_val.timestamp())
                pnl = 0.0
                for inst_id, (act, qty, mult, op) in leg_map.items():
                    cur = current_prices.get(inst_id, op)
                    pnl += (op - cur) * qty * mult if act == "S" else (cur - op) * qty * mult
                after_exit_pts.append({"time": ts_int, "value": round(pnl, 2)})

    # ── Entry / exit markers ───────────────────────────────────────────────────
    net_pnl = _nan_to_none(pos.get("net_pnl")) or 0.0

    if exit_reason == "expiry":
        exit_color = _GREEN if net_pnl >= 0 else _RED
        exit_label = "Exp"
    elif exit_reason == "take_profit":
        exit_color = _GREEN
        exit_label = "TP"
    elif exit_reason == "stop_loss":
        exit_color = _RED
        exit_label = "SL"
    else:
        exit_color = _AMBER
        exit_label = exit_reason or "Exit"

    exit_text = (
        f"Exp {exit_mark:.2f}"
        if exit_reason == "expiry"
        else f"Exit {exit_mark:.2f} – {exit_label}"
    )

    # In leg mode use the leg's own entry/exit prices for marker labels.
    if leg_id and active_legs:
        lg = active_legs[0]
        entry_lbl = (
            f"Entry {lg['open_price']:.2f}" if lg.get("open_price") else f"Entry {open_mark:.2f}"
        )
        ep = lg.get("exit_price")
        if exit_reason == "expiry":
            exit_lbl = f"Exp {ep:.2f}" if ep else "Exp"
        else:
            exit_lbl = f"Exit {ep:.2f} – {exit_label}" if ep else exit_label
    else:
        entry_lbl = f"Entry {open_mark:.2f}"
        exit_lbl = exit_text

    markers = [
        {
            "time": open_ts,
            "position": "belowBar",
            "color": _GREEN,
            "shape": "arrowUp",
            "text": entry_lbl,
        },
        {
            "time": exit_ts,
            "position": "aboveBar",
            "color": exit_color,
            "shape": "arrowDown",
            "text": exit_lbl,
        },
    ]

    return JSONResponse(
        {
            "has_data": True,
            "candles": candles,
            "volume": volume,
            "markers": markers,
            "be_lines": be_lines,
            "strike_lines": strike_lines,
            "pnl": pnl_pts if len(pnl_pts) > 1 else [],
            "after_exit": after_exit_pts if len(after_exit_pts) > 1 else [],
            "tp_sl_lines": tp_sl_lines,
            "open_ts": open_ts,
            "exit_ts": exit_ts,
            "leg_mode": leg_id is not None,
        }
    )
