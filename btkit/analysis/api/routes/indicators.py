"""Indicator overlay endpoints for per-trade charts (requires BTKIT_INPUT_DB)."""

from __future__ import annotations

from datetime import UTC

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..db import query as out_query
from ..db_input import connect as input_connect

router = APIRouter()
UTC = UTC

_POS_SQL = """
SELECT p.open_time, p.exit_time, pl.instrument_id
FROM position p
JOIN position_leg pl ON pl.position_id = p.id
WHERE p.id = ?
ORDER BY pl.id
LIMIT 1
"""


def _get_underlying_id(con, instrument_id: int) -> int | None:
    row = con.execute(
        "SELECT underlying_id FROM option_bars WHERE instrument_id = ? LIMIT 1",
        [instrument_id],
    ).fetchone()
    return row[0] if row else None


def _day_window(open_time, exit_time):
    day_start = open_time.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
    day_end = exit_time.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=UTC)
    return day_start, day_end


@router.get("/positions/{position_id}/indicators")
def list_indicators(position_id: int) -> JSONResponse:
    """
    Return available indicators for the position's underlying.

    Response:
        { "indicators": [{ "id": int, "name": str }, ...] }

    Returns an empty list when no input DB is configured or no indicators
    have been computed for this underlying.
    """
    con = input_connect()
    if con is None:
        return JSONResponse({"indicators": []})

    try:
        pos_cols, pos_rows = out_query(_POS_SQL, [position_id])
        if not pos_rows:
            raise HTTPException(status_code=404, detail="Position not found")

        pos = dict(zip(pos_cols, pos_rows[0], strict=False))
        underlying_id = _get_underlying_id(con, pos["instrument_id"])
        if underlying_id is None:
            return JSONResponse({"indicators": []})

        rows = con.execute(
            "SELECT id, name FROM indicator_definition WHERE underlying_id = ? ORDER BY name",
            [underlying_id],
        ).fetchall()

        return JSONResponse({"indicators": [{"id": r[0], "name": r[1]} for r in rows]})
    finally:
        con.close()


@router.get("/positions/{position_id}/indicators/{indicator_id}")
def get_indicator_data(position_id: int, indicator_id: int) -> JSONResponse:
    """
    Return time-series data for one indicator over the position's full day window.

    Response: [{ "time": unix_seconds, "value": float }, ...]
    """
    con = input_connect()
    if con is None:
        return JSONResponse([])

    try:
        pos_cols, pos_rows = out_query(_POS_SQL, [position_id])
        if not pos_rows:
            raise HTTPException(status_code=404, detail="Position not found")

        pos = dict(zip(pos_cols, pos_rows[0], strict=False))
        open_time = pos["open_time"]
        exit_time = pos["exit_time"]
        if not open_time or not exit_time:
            return JSONResponse([])

        day_start, day_end = _day_window(open_time, exit_time)

        rows = con.execute(
            """
            SELECT ts_event, value
            FROM indicator_bars
            WHERE indicator_id = ? AND ts_event >= ? AND ts_event <= ?
            ORDER BY ts_event
            """,
            [indicator_id, day_start, day_end],
        ).fetchall()

        return JSONResponse(
            [{"time": int(r[0].timestamp()), "value": r[1]} for r in rows if r[1] is not None]
        )
    finally:
        con.close()
