"""
Chart Explorer endpoints — free-form browsing of input-DB market data.

Serves the /explore dashboard page: list underlying contracts (individual + a
synthetic front-month continuous series per root), aggregate 1-minute
underlying_bars to a requested timeframe, and expose indicator overlays.

All endpoints require BTKIT_INPUT_DB; without it they return empty payloads so
the page degrades gracefully (same convention as the chart/indicator routes).

Aggregation is timezone-aware: bars are bucketed on the session-local calendar
(America/New_York) so daily/weekly/monthly candles align to trading days rather
than UTC midnight. Times are emitted as unix seconds (the bucket-start instant).
"""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime

import polars as pl
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from btkit.db.input_db import InputDatabase

from ..db import query as out_query
from ..db_input import connect as input_connect

router = APIRouter()

SESSION_TZ = "America/New_York"

# Chart timeframe → Polars group_by_dynamic `every` string.
TF_EVERY: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1H": "1h",
    "1D": "1d",
    "1W": "1w",
    "1M": "1mo",
}

_ROOT_RE = re.compile(r"^([A-Z]+?)[FGHJKMNQUVXZ]\d{1,2}$")

# Indicators whose values are NOT in price units go to a separate panel;
# everything else overlays on the price scale.
_PANEL_KEYWORDS = (
    "rsi", "atr", "adx", "stoch", "macd", "mom", "roc", "cci", "willr",
    "obv", "volatility", "zscore", "z_score", "ves", "percentile", "rank",
)

_UP = "rgba(74,222,128,0.35)"
_DOWN = "rgba(248,113,113,0.35)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _root_of(symbol: str) -> str | None:
    """Contract root from a futures symbol, e.g. 'ESM6' → 'ES'. None if no match."""
    m = _ROOT_RE.match(symbol or "")
    return m.group(1) if m else None


def _placement(name: str) -> str:
    n = (name or "").lower()
    return "panel" if any(k in n for k in _PANEL_KEYWORDS) else "overlay"


def _parse_date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def _range_dt(start: date | None, end: date | None) -> tuple[datetime | None, datetime | None]:
    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC) if start else None
    end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC) if end else None
    return start_dt, end_dt


def _to_session_tz(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise ts_event to the session timezone (instant preserved)."""
    return df.with_columns(pl.col("ts_event").dt.convert_time_zone(SESSION_TZ))


def _aggregate_ohlcv(df: pl.DataFrame, timeframe: str) -> tuple[list, list]:
    """Bucket 1-minute OHLCV to `timeframe`; return (candles, volume)."""
    every = TF_EVERY[timeframe]
    agg = (
        _to_session_tz(df)
        .sort("ts_event")
        .group_by_dynamic("ts_event", every=every, closed="left", label="left")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum().alias("volume"),
        )
        .sort("ts_event")
        .with_columns(pl.col("ts_event").dt.epoch("s").alias("time"))
    )
    candles = [
        {"time": int(t), "open": float(o), "high": float(h), "low": float(lo), "close": float(c)}
        for t, o, h, lo, c in zip(
            agg["time"], agg["open"], agg["high"], agg["low"], agg["close"], strict=False
        )
    ]
    volume = [
        {"time": int(t), "value": float(v or 0), "color": _UP if c >= o else _DOWN}
        for t, v, o, c in zip(
            agg["time"], agg["volume"], agg["open"], agg["close"], strict=False
        )
    ]
    return candles, volume


def _bucket_last(df: pl.DataFrame, timeframe: str) -> list:
    """Bucket an indicator (ts_event, value) series to `timeframe`, last-in-bucket."""
    if df.is_empty():
        return []
    every = TF_EVERY[timeframe]
    agg = (
        _to_session_tz(df.filter(pl.col("value").is_not_null()))
        .sort("ts_event")
        .group_by_dynamic("ts_event", every=every, closed="left", label="left")
        .agg(pl.col("value").last())
        .sort("ts_event")
        .with_columns(pl.col("ts_event").dt.epoch("s").alias("time"))
        .filter(pl.col("value").is_not_null())
    )
    return [
        {"time": int(t), "value": float(v)}
        for t, v in zip(agg["time"], agg["value"], strict=False)
    ]


def _front_month_bars(idb: InputDatabase, root: str, start: date | None, end: date | None):
    """Stitched front-month underlying bars for a root over [start, end]."""
    row = idb._con.execute(
        "SELECT MIN(ts_event), MAX(ts_event) FROM underlying_bars WHERE symbol LIKE ?",
        [f"{root}%"],
    ).fetchone()
    if not row or row[0] is None:
        return pl.DataFrame(), None, None
    lo = start or row[0].astimezone(UTC).date()
    hi = end or row[1].astimezone(UTC).date()
    start_dt = datetime(lo.year, lo.month, lo.day, tzinfo=UTC)
    end_dt = datetime(hi.year, hi.month, hi.day, 23, 59, 59, tzinfo=UTC)

    schedule = idb.front_future_schedule(root, lo, hi)
    bars = idb.underlying_bars_for_root(root, start_dt, end_dt)
    if schedule.is_empty() or bars.is_empty():
        return pl.DataFrame(), lo, hi

    df = (
        bars.with_columns(pl.col("ts_event").dt.date().alias("_d"))
        .join(schedule, left_on="_d", right_on="date", how="left")
        .filter(pl.col("instrument_id") == pl.col("underlying_id"))
        .select(["ts_event", "open", "high", "low", "close", "volume"])
    )
    return df, lo, hi


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/explore/contracts")
def list_contracts() -> JSONResponse:
    """
    Available contracts for the explorer.

    Response: { "contracts": [ {kind, symbol, ...}, ... ] } with continuous
    front-month roots first, then individual contracts.
    """
    con = input_connect()
    if con is None:
        return JSONResponse({"contracts": []})
    try:
        rows = con.execute(
            "SELECT instrument_id, symbol, MIN(ts_event), MAX(ts_event), COUNT(*) "
            "FROM underlying_bars GROUP BY instrument_id, symbol ORDER BY symbol"
        ).fetchall()

        contracts: list[dict] = []
        roots: dict[str, list] = {}
        for iid, sym, mn, mx, n in rows:
            contracts.append(
                {
                    "kind": "contract",
                    "instrument_id": int(iid),
                    "symbol": sym,
                    "start": mn.date().isoformat(),
                    "end": mx.date().isoformat(),
                    "bars": int(n),
                }
            )
            r = _root_of(sym)
            if r:
                agg = roots.setdefault(r, [mn, mx, 0])
                agg[0] = min(agg[0], mn)
                agg[1] = max(agg[1], mx)
                agg[2] += n

        continuous = [
            {
                "kind": "continuous",
                "root": r,
                "symbol": r,
                "start": v[0].date().isoformat(),
                "end": v[1].date().isoformat(),
                "bars": int(v[2]),
            }
            for r, v in sorted(roots.items())
        ]
        return JSONResponse({"contracts": continuous + contracts})
    finally:
        con.close()


@router.get("/explore/bars")
def get_bars(
    timeframe: str = Query(...),
    instrument_id: int | None = Query(None),
    root: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> JSONResponse:
    """Aggregated OHLCV candles + volume for a contract at `timeframe`."""
    if timeframe not in TF_EVERY:
        raise HTTPException(status_code=400, detail=f"unknown timeframe: {timeframe}")
    if instrument_id is None and not root:
        raise HTTPException(status_code=400, detail="instrument_id or root is required")

    con = input_connect()
    if con is None:
        return JSONResponse({"has_data": False, "candles": [], "volume": []})
    try:
        d0, d1 = _parse_date(start), _parse_date(end)
        if instrument_id is not None:
            start_dt, end_dt = _range_dt(d0, d1)
            clauses = ["instrument_id = ?"]
            params: list = [instrument_id]
            if start_dt is not None:
                clauses.append("ts_event >= ?")
                params.append(start_dt)
            if end_dt is not None:
                clauses.append("ts_event <= ?")
                params.append(end_dt)
            df = con.execute(
                "SELECT ts_event, open, high, low, close, volume FROM underlying_bars "
                f"WHERE {' AND '.join(clauses)} ORDER BY ts_event",
                params,
            ).pl()
        else:
            path = os.environ.get("BTKIT_INPUT_DB")
            idb = InputDatabase(path)
            try:
                df, _, _ = _front_month_bars(idb, root, d0, d1)
            finally:
                idb._con.close()

        if df.is_empty():
            return JSONResponse({"has_data": False, "candles": [], "volume": []})

        candles, volume = _aggregate_ohlcv(df, timeframe)
        return JSONResponse({"has_data": True, "candles": candles, "volume": volume})
    finally:
        con.close()


@router.get("/explore/indicators")
def list_explore_indicators(
    instrument_id: int | None = Query(None),
    root: str | None = Query(None),
) -> JSONResponse:
    """
    Indicators available for a contract.

    For an individual contract each entry has a concrete `id`; for a continuous
    root the id is null (indicators are keyed by `name` and stitched per date).
    Response: { "indicators": [ {id, name, placement}, ... ] }
    """
    con = input_connect()
    if con is None:
        return JSONResponse({"indicators": []})
    try:
        if instrument_id is not None:
            rows = con.execute(
                "SELECT id, name FROM indicator_definition WHERE underlying_id = ? ORDER BY name",
                [instrument_id],
            ).fetchall()
            inds = [
                {"id": int(r[0]), "name": r[1], "placement": _placement(r[1])} for r in rows
            ]
        elif root:
            rows = con.execute(
                "SELECT DISTINCT d.name FROM indicator_definition d "
                "JOIN underlying_bars u ON u.instrument_id = d.underlying_id "
                "WHERE u.symbol LIKE ? ORDER BY d.name",
                [f"{root}%"],
            ).fetchall()
            inds = [{"id": None, "name": r[0], "placement": _placement(r[0])} for r in rows]
        else:
            inds = []
        return JSONResponse({"indicators": inds})
    finally:
        con.close()


@router.get("/explore/indicator")
def get_indicator(
    timeframe: str = Query(...),
    indicator_id: int | None = Query(None),
    name: str | None = Query(None),
    root: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> JSONResponse:
    """
    One indicator series bucketed to `timeframe`.

    Individual contract: pass indicator_id. Continuous root: pass name + root
    (the front-month contract's indicator of that name is used per date).
    Response: { "placement": str, "data": [ {time, value}, ... ] }
    """
    if timeframe not in TF_EVERY:
        raise HTTPException(status_code=400, detail=f"unknown timeframe: {timeframe}")

    con = input_connect()
    if con is None:
        return JSONResponse({"placement": "overlay", "data": []})
    try:
        d0, d1 = _parse_date(start), _parse_date(end)

        if indicator_id is not None:
            meta = con.execute(
                "SELECT name FROM indicator_definition WHERE id = ?", [indicator_id]
            ).fetchone()
            placement = _placement(meta[0]) if meta else "overlay"
            start_dt, end_dt = _range_dt(d0, d1)
            clauses = ["indicator_id = ?"]
            params: list = [indicator_id]
            if start_dt is not None:
                clauses.append("ts_event >= ?")
                params.append(start_dt)
            if end_dt is not None:
                clauses.append("ts_event <= ?")
                params.append(end_dt)
            df = con.execute(
                "SELECT ts_event, value FROM indicator_bars "
                f"WHERE {' AND '.join(clauses)} ORDER BY ts_event",
                params,
            ).pl()
            return JSONResponse({"placement": placement, "data": _bucket_last(df, timeframe)})

        if not name or not root:
            raise HTTPException(status_code=400, detail="indicator_id, or name + root, is required")

        # Continuous: stitch the front-month contract's indicator of this name.
        path = os.environ.get("BTKIT_INPUT_DB")
        idb = InputDatabase(path)
        try:
            _, lo, hi = _front_month_bars(idb, root, d0, d1)
            if lo is None:
                return JSONResponse({"placement": _placement(name), "data": []})
            schedule = idb.front_future_schedule(root, lo, hi)
            if schedule.is_empty():
                return JSONResponse({"placement": _placement(name), "data": []})
            start_dt = datetime(lo.year, lo.month, lo.day, tzinfo=UTC)
            end_dt = datetime(hi.year, hi.month, hi.day, 23, 59, 59, tzinfo=UTC)
            raw = idb._con.execute(
                """
                SELECT ib.ts_event, ib.value, d.underlying_id
                FROM indicator_bars ib
                JOIN indicator_definition d ON d.id = ib.indicator_id
                WHERE d.name = ?
                  AND d.underlying_id IN (
                      SELECT DISTINCT instrument_id FROM underlying_bars WHERE symbol LIKE ?
                  )
                  AND ib.ts_event >= ? AND ib.ts_event <= ?
                ORDER BY ib.ts_event
                """,
                [name, f"{root}%", start_dt, end_dt],
            ).pl()
        finally:
            idb._con.close()

        if raw.is_empty():
            return JSONResponse({"placement": _placement(name), "data": []})

        stitched = (
            raw.with_columns(pl.col("ts_event").dt.date().alias("_d"))
            .join(
                schedule.select(["date", "underlying_id"]),
                left_on="_d",
                right_on="date",
                how="left",
            )
            .filter(pl.col("underlying_id") == pl.col("underlying_id_right"))
            .select(["ts_event", "value"])
        )
        data = _bucket_last(stitched, timeframe)
        return JSONResponse({"placement": _placement(name), "data": data})
    finally:
        con.close()


# Marker / equity overlay colors (match the dashboard theme).
_MK_UP = "#4ade80"
_MK_DOWN = "#f87171"


@router.get("/explore/overlay")
def get_overlay(
    backtest_id: int = Query(...),
    instrument_id: int | None = Query(None),
    root: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> JSONResponse:
    """
    Backtest overlay for the currently-viewed contract: entry/exit markers placed
    at the underlying price, plus a cumulative realized-P&L series for a sub-panel.

    Only positions whose underlying matches the viewed contract (exact
    instrument_id, or symbol-prefix for a continuous root) and whose life overlaps
    the [start, end] window are included. Requires BTKIT_INPUT_DB; the position/leg
    rows come from the output DB.
    """
    con = input_connect()
    if con is None:
        return JSONResponse({"markers": [], "equity": [], "n_positions": 0})
    try:
        cols, rows = out_query(
            "SELECT p.id, p.open_time, p.exit_time, p.exit_reason, p.net_pnl, "
            "       MIN(pl.instrument_id) AS leg_iid "
            "FROM position p JOIN position_leg pl ON pl.position_id = p.id "
            "WHERE p.backtest_id = ? "
            "GROUP BY p.id, p.open_time, p.exit_time, p.exit_reason, p.net_pnl "
            "ORDER BY p.open_time",
            [backtest_id],
        )
        positions = [dict(zip(cols, r, strict=False)) for r in rows]
        if not positions:
            return JSONResponse({"markers": [], "equity": [], "n_positions": 0})

        # Resolve each position's option leg → underlying_id, and underlying → symbol.
        leg_ids = sorted({int(p["leg_iid"]) for p in positions if p["leg_iid"] is not None})
        opt_to_und: dict[int, int] = {}
        if leg_ids:
            id_list = ",".join(str(i) for i in leg_ids)
            opt_to_und = {
                int(iid): int(uid)
                for iid, uid in con.execute(
                    f"SELECT DISTINCT instrument_id, underlying_id FROM option_bars "
                    f"WHERE instrument_id IN ({id_list})"
                ).fetchall()
            }
        _sym_rows = con.execute(
            "SELECT DISTINCT instrument_id, symbol FROM underlying_bars"
        ).fetchall()
        und_syms = {int(i): s for i, s in _sym_rows}

        def _matches(p: dict) -> bool:
            iid = p["leg_iid"]
            uid = opt_to_und.get(int(iid)) if iid is not None else None
            if uid is None:
                return False
            if instrument_id is not None:
                return uid == instrument_id
            if root:
                return (und_syms.get(uid) or "").startswith(root)
            return True

        s_dt, e_dt = _range_dt(_parse_date(start), _parse_date(end))

        def _in_range(p: dict) -> bool:
            ot, xt = p["open_time"], p["exit_time"]
            if s_dt is not None and xt is not None and xt < s_dt:
                return False
            if e_dt is not None and ot is not None and ot > e_dt:
                return False
            return True

        kept = [p for p in positions if _matches(p) and _in_range(p)]

        markers: list[dict] = []
        equity: list[dict] = []
        cum = 0.0
        for p in sorted(kept, key=lambda x: (x["exit_time"] or x["open_time"])):
            ot, xt = p["open_time"], p["exit_time"]
            pnl = float(p["net_pnl"] or 0.0)
            if ot is not None:
                markers.append(
                    {"time": int(ot.timestamp()), "position": "belowBar",
                     "color": _MK_UP, "shape": "arrowUp", "text": ""}
                )
            if xt is not None:
                markers.append(
                    {"time": int(xt.timestamp()), "position": "aboveBar",
                     "color": _MK_UP if pnl >= 0 else _MK_DOWN, "shape": "arrowDown",
                     "text": f"{'+' if pnl >= 0 else ''}{pnl:.0f}"}
                )
                cum += pnl
                equity.append({"time": int(xt.timestamp()), "value": round(cum, 2)})

        markers.sort(key=lambda m: m["time"])
        return JSONResponse({"markers": markers, "equity": equity, "n_positions": len(kept)})
    finally:
        con.close()
