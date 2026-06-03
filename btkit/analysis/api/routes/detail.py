"""Per-study and per-backtest detail endpoints."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..db import query

router = APIRouter()

# ── Shared helpers ────────────────────────────────────────────────────────────

EXIT_REASON_MAP = {"take_profit": "TP", "stop_loss": "SL", "expiry": "Expiry"}


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def _row_to_dict(cols: list[str], row: tuple) -> dict:
    return {c: _clean(v) for c, v in zip(cols, row)}


def _extract_params(params_json: str | None) -> dict:
    if not params_json:
        return {}
    try:
        p = json.loads(params_json)
        result: dict[str, Any] = {}
        for trade in p.get("trades", []):
            legs = trade.get("legs", [])
            short = next(
                (l for l in legs if "sell" in l.get("action", "")),
                legs[0] if legs else {},
            )
            result["delta"]           = short.get("delta")
            ex = trade.get("exit", {})
            result["take_profit_pct"] = ex.get("take_profit_pct")
            result["stop_loss"]       = ex.get("stop_loss")
            result["min_credit"]      = trade.get("entry", {}).get("min_credit")
            break
    except Exception:
        return {}
    return result


def _strategy_label(name: str) -> str:
    return name.replace("_", " ").title()


def _detect_sweep_axes(params_list: list[dict]) -> list[str]:
    if not params_list:
        return []
    all_keys: set[str] = set().union(*[set(p.keys()) for p in params_list])
    return sorted(k for k in all_keys if len({str(p.get(k)) for p in params_list}) > 1)


# ── Study detail SQL ──────────────────────────────────────────────────────────

_STUDY_ROW_SQL = """
SELECT
    s.id, s.name, s.note, s.total_combinations, s.created_at, s.finished_at,
    COUNT(b.id)                                          AS n_backtests,
    COUNT(CASE WHEN b.status = 'completed' THEN 1 END)  AS n_completed,
    COUNT(CASE WHEN b.status = 'error'     THEN 1 END)  AS n_failed,
    COUNT(CASE WHEN b.status = 'running'   THEN 1 END)  AS n_running
FROM study s
LEFT JOIN backtest b ON b.study_id = s.id
WHERE s.id = ?
GROUP BY s.id, s.name, s.note, s.total_combinations, s.created_at, s.finished_at
"""

_STUDY_BACKTESTS_SQL = """
WITH
bt_ids AS (
    SELECT id, combination_id, strategy_name, strategy_params, initial_equity, status, created_at
    FROM backtest WHERE study_id = ?
),
completed_ids AS (SELECT id FROM bt_ids WHERE status = 'completed'),
date_ranges AS (
    SELECT p.backtest_id,
        CAST(MIN(p.open_time) AS DATE) AS start_date,
        CAST(MAX(p.exit_time)  AS DATE) AS end_date
    FROM position p INNER JOIN completed_ids c ON c.id = p.backtest_id
    GROUP BY p.backtest_id
),
daily_actives AS (
    SELECT p.backtest_id, CAST(p.open_time AS DATE) AS trade_date, SUM(p.net_pnl) AS daily_pnl
    FROM position p INNER JOIN completed_ids c ON c.id = p.backtest_id
    GROUP BY p.backtest_id, CAST(p.open_time AS DATE)
),
all_weekdays AS (
    SELECT dr.backtest_id, cal.d::DATE AS trade_date
    FROM date_ranges dr
    CROSS JOIN LATERAL (
        SELECT unnest(generate_series(dr.start_date, dr.end_date, INTERVAL '1 day')) AS d
    ) cal
    WHERE dayofweek(cal.d::DATE) NOT IN (0, 6)
),
daily AS (
    SELECT w.backtest_id, COALESCE(a.daily_pnl, 0.0) AS daily_pnl
    FROM all_weekdays w
    LEFT JOIN daily_actives a
           ON a.backtest_id = w.backtest_id AND a.trade_date = w.trade_date
),
daily_stats AS (
    SELECT backtest_id,
        AVG(daily_pnl)         AS mean_daily_pnl,
        STDDEV_SAMP(daily_pnl) AS std_daily_pnl
    FROM daily GROUP BY backtest_id
),
trade_stats AS (
    SELECT
        p.backtest_id,
        COUNT(*) AS n_trades,
        SUM(p.net_pnl) AS total_pnl,
        AVG(p.net_pnl) AS avg_pnl,
        CAST(COUNT(CASE WHEN p.net_pnl > 0 THEN 1 END) AS DOUBLE)
            / NULLIF(COUNT(*), 0)                               AS win_rate,
        SUM(CASE WHEN p.net_pnl > 0 THEN p.net_pnl ELSE 0.0 END)   AS gross_win,
        -SUM(CASE WHEN p.net_pnl < 0 THEN p.net_pnl ELSE 0.0 END)  AS gross_loss,
        AVG(DATEDIFF('minute', p.open_time, p.exit_time))       AS avg_duration_min,
        CAST(MIN(p.open_time) AS DATE)                          AS start_date,
        CAST(MAX(p.exit_time)  AS DATE)                         AS end_date
    FROM position p INNER JOIN completed_ids c ON c.id = p.backtest_id
    GROUP BY p.backtest_id
)
SELECT
    b.id,
    b.combination_id,
    b.strategy_name,
    b.strategy_params,
    b.initial_equity,
    b.status,
    b.created_at,
    COALESCE(t.n_trades, 0)     AS n_trades,
    COALESCE(t.total_pnl, 0.0)  AS total_pnl,
    t.avg_pnl,
    t.win_rate,
    t.avg_duration_min,
    t.start_date,
    t.end_date,
    CASE WHEN COALESCE(d.std_daily_pnl, 0.0) > 0
        THEN (d.mean_daily_pnl / d.std_daily_pnl) * SQRT(252.0)
        ELSE NULL END                                           AS sharpe,
    CASE WHEN COALESCE(b.initial_equity, 0) > 0
        THEN (COALESCE(t.total_pnl, 0.0) / b.initial_equity) * 100.0
        ELSE NULL END                                           AS total_return_pct,
    CASE WHEN COALESCE(t.gross_loss, 0.0) > 0
        THEN t.gross_win / t.gross_loss
        ELSE NULL END                                           AS profit_factor
FROM bt_ids b
LEFT JOIN trade_stats  t ON t.backtest_id = b.id
LEFT JOIN daily_stats  d ON d.backtest_id = b.id
ORDER BY b.combination_id
"""

_EQUITY_SQL = """
SELECT
    p.backtest_id,
    b.initial_equity,
    CAST(p.exit_time AS DATE) AS exit_date,
    SUM(p.net_pnl) OVER (
        PARTITION BY p.backtest_id
        ORDER BY p.open_time, p.id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cum_pnl
FROM position p
JOIN backtest b ON b.id = p.backtest_id
WHERE b.study_id = ? AND b.status = 'completed'
ORDER BY p.backtest_id, p.open_time, p.id
"""

_POSITIONS_SQL = """
SELECT
    ROW_NUMBER() OVER (ORDER BY p.open_time, p.id)  AS trade_num,
    p.id,
    p.exit_reason,
    CAST(p.open_time AS DATE)                        AS open_date,
    CAST(p.exit_time  AS DATE)                       AS exit_date,
    p.net_pnl,
    DATEDIFF('minute', p.open_time, p.exit_time)     AS duration_min,
    SUM(p.net_pnl) OVER (
        ORDER BY p.open_time, p.id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                AS cum_pnl,
    b.initial_equity
FROM position p
JOIN backtest b ON b.id = p.backtest_id
WHERE p.backtest_id = ?
ORDER BY p.open_time, p.id
"""

# Single-backtest detail (reuses same CTE structure, filtered to one backtest)
_BACKTEST_DETAIL_SQL = """
WITH
dr AS (
    SELECT
        CAST(MIN(open_time) AS DATE) AS start_date,
        CAST(MAX(exit_time)  AS DATE) AS end_date
    FROM position WHERE backtest_id = ?
),
daily_actives AS (
    SELECT CAST(open_time AS DATE) AS trade_date, SUM(net_pnl) AS daily_pnl
    FROM position WHERE backtest_id = ?
    GROUP BY CAST(open_time AS DATE)
),
all_weekdays AS (
    SELECT cal.d::DATE AS trade_date
    FROM dr
    CROSS JOIN LATERAL (
        SELECT unnest(generate_series(dr.start_date, dr.end_date, INTERVAL '1 day')) AS d
    ) cal
    WHERE dayofweek(cal.d::DATE) NOT IN (0, 6)
),
daily AS (
    SELECT w.trade_date, COALESCE(a.daily_pnl, 0.0) AS daily_pnl
    FROM all_weekdays w
    LEFT JOIN daily_actives a ON a.trade_date = w.trade_date
),
daily_stats AS (
    SELECT AVG(daily_pnl) AS mean_daily_pnl, STDDEV_SAMP(daily_pnl) AS std_daily_pnl FROM daily
),
trade_stats AS (
    SELECT
        COUNT(*) AS n_trades,
        SUM(net_pnl) AS total_pnl,
        AVG(net_pnl) AS avg_pnl,
        CAST(COUNT(CASE WHEN net_pnl > 0 THEN 1 END) AS DOUBLE) / NULLIF(COUNT(*), 0) AS win_rate,
        SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0.0 END)   AS gross_win,
        -SUM(CASE WHEN net_pnl < 0 THEN net_pnl ELSE 0.0 END)  AS gross_loss,
        AVG(CASE WHEN net_pnl > 0 THEN net_pnl ELSE NULL END)   AS avg_win,
        AVG(CASE WHEN net_pnl < 0 THEN net_pnl ELSE NULL END)   AS avg_loss,
        COUNT(CASE WHEN exit_reason = 'take_profit' THEN 1 END) AS n_take_profit,
        COUNT(CASE WHEN exit_reason = 'stop_loss'   THEN 1 END) AS n_stop_loss,
        AVG(DATEDIFF('minute', open_time, exit_time))           AS avg_duration_min,
        CAST(MIN(open_time) AS DATE)                            AS start_date,
        CAST(MAX(exit_time)  AS DATE)                           AS end_date
    FROM position WHERE backtest_id = ?
)
SELECT
    b.id, b.study_id, b.combination_id, b.strategy_name, b.strategy_params,
    b.initial_equity, b.status, b.created_at,
    COALESCE(t.n_trades, 0)    AS n_trades,
    COALESCE(t.total_pnl, 0.0) AS total_pnl,
    t.avg_pnl, t.win_rate, t.avg_win, t.avg_loss,
    t.n_take_profit, t.n_stop_loss,
    t.avg_duration_min, t.start_date, t.end_date,
    CASE WHEN COALESCE(d.std_daily_pnl, 0.0) > 0
        THEN (d.mean_daily_pnl / d.std_daily_pnl) * SQRT(252.0)
        ELSE NULL END AS sharpe,
    CASE WHEN COALESCE(b.initial_equity, 0) > 0
        THEN (COALESCE(t.total_pnl, 0.0) / b.initial_equity) * 100.0
        ELSE NULL END AS total_return_pct,
    CASE WHEN COALESCE(t.gross_loss, 0.0) > 0
        THEN t.gross_win / t.gross_loss
        ELSE NULL END AS profit_factor
FROM backtest b
CROSS JOIN trade_stats t
CROSS JOIN daily_stats d
WHERE b.id = ?
"""


# ── Position detail SQL ───────────────────────────────────────────────────────

_POSITION_DETAIL_SQL = """
SELECT
    p.id,
    p.backtest_id,
    p.trade_name,
    p.open_time,
    p.exit_time,
    p.exit_reason,
    DATEDIFF('minute', p.open_time, p.exit_time) AS duration_min,
    p.net_pnl,
    p.open_mark,
    b.strategy_name,
    b.strategy_params,
    b.initial_equity
FROM position p
JOIN backtest b ON b.id = p.backtest_id
WHERE p.id = ?
"""

_POSITION_LEGS_SQL = """
SELECT
    l.id,
    l.instrument_id,
    l.symbol,
    CAST(l.expiration AS VARCHAR) AS expiration,
    l.strike_price,
    l.right,
    l.action,
    l.quantity,
    l.multiplier,
    l.open_price,
    l.exit_price,
    l.entry_delta,
    l.entry_iv,
    l.entry_gamma,
    l.entry_theta,
    l.entry_vega,
    l.entry_dte
FROM position_leg l
WHERE l.position_id = ?
ORDER BY l.id
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/studies/{study_id}")
def study_detail(study_id: int) -> JSONResponse:
    # Study row
    s_cols, s_rows = query(_STUDY_ROW_SQL, [study_id])
    if not s_rows:
        raise HTTPException(status_code=404, detail="Study not found")
    study = _row_to_dict(s_cols, s_rows[0])

    # All backtests with metrics
    bt_cols, bt_rows = query(_STUDY_BACKTESTS_SQL, [study_id])
    backtests = []
    params_list: list[dict] = []
    strategies: set[str] = set()
    best_sharpe: float | None = None
    best_return: float | None = None
    total_trades = 0
    data_start: str | None = None
    data_end: str | None = None

    for row in bt_rows:
        d = _row_to_dict(bt_cols, row)
        params = _extract_params(d.pop("strategy_params"))
        d["params"]          = params
        d["strategy_label"]  = _strategy_label(d["strategy_name"])
        backtests.append(d)

        if d["status"] == "completed":
            params_list.append(params)
            strategies.add(d["strategy_name"])
            total_trades += d["n_trades"]
            if d["sharpe"] is not None:
                if best_sharpe is None or d["sharpe"] > best_sharpe:
                    best_sharpe = round(d["sharpe"], 2)
            if d["total_return_pct"] is not None:
                if best_return is None or d["total_return_pct"] > best_return:
                    best_return = round(d["total_return_pct"], 2)
            if d["start_date"] is not None:
                data_start = d["start_date"] if data_start is None else min(data_start, d["start_date"])
            if d["end_date"] is not None:
                data_end = d["end_date"] if data_end is None else max(data_end, d["end_date"])

    strats = sorted(strategies)
    sweep_axes = _detect_sweep_axes(params_list)

    return JSONResponse({
        **study,
        "strategies":      strats,
        "strategy_labels": [_strategy_label(n) for n in strats],
        "sweep_axes":      sweep_axes,
        "data_start":      data_start,
        "data_end":        data_end,
        "total_trades":    total_trades,
        "best_sharpe":     best_sharpe,
        "best_return_pct": best_return,
        "backtests":       backtests,
    })


@router.get("/studies/{study_id}/equity")
def study_equity(study_id: int) -> JSONResponse:
    cols, rows = query(_EQUITY_SQL, [study_id])
    curves: dict[int, dict] = {}
    for row in rows:
        d = _row_to_dict(cols, row)
        bid = d["backtest_id"]
        if bid not in curves:
            curves[bid] = {
                "backtest_id":    bid,
                "initial_equity": d["initial_equity"] or 100_000,
                "cum_pnl":        [],
                "exit_dates":     [],
            }
        curves[bid]["cum_pnl"].append(d["cum_pnl"])
        curves[bid]["exit_dates"].append(d["exit_date"])
    return JSONResponse(list(curves.values()))


@router.get("/backtests/{backtest_id}")
def backtest_detail(backtest_id: int) -> JSONResponse:
    cols, rows = query(_BACKTEST_DETAIL_SQL, [backtest_id, backtest_id, backtest_id, backtest_id])
    if not rows:
        raise HTTPException(status_code=404, detail="Backtest not found")
    d = _row_to_dict(cols, rows[0])
    d["params"]         = _extract_params(d.pop("strategy_params"))
    d["strategy_label"] = _strategy_label(d["strategy_name"])
    return JSONResponse(d)


@router.get("/backtests/{backtest_id}/positions")
def backtest_positions(backtest_id: int) -> JSONResponse:
    cols, rows = query(_POSITIONS_SQL, [backtest_id])
    result = []
    for row in rows:
        d = _row_to_dict(cols, row)
        d["exit_reason"] = EXIT_REASON_MAP.get(d["exit_reason"], d["exit_reason"])
        d["return_pct"]  = (
            round(d["net_pnl"] / d["initial_equity"] * 100, 4)
            if d["initial_equity"] else None
        )
        d.pop("initial_equity", None)
        result.append(d)
    return JSONResponse(result)


@router.get("/positions/{position_id}")
def position_detail(position_id: int) -> JSONResponse:
    cols, rows = query(_POSITION_DETAIL_SQL, [position_id])
    if not rows:
        raise HTTPException(status_code=404, detail="Position not found")
    d = _row_to_dict(cols, rows[0])

    params = _extract_params(d.pop("strategy_params"))
    d["exit_reason"]    = EXIT_REASON_MAP.get(d["exit_reason"], d["exit_reason"])
    d["strategy_label"] = _strategy_label(d["strategy_name"])
    d["params"]         = params

    # Legs
    leg_cols, leg_rows = query(_POSITION_LEGS_SQL, [position_id])
    legs = [_row_to_dict(leg_cols, r) for r in leg_rows]
    d["legs"] = legs

    # Net credit received (positive = credit spread, negative = debit spread)
    credit = sum(
        (1 if leg["action"] == "STO" else -1)
        * (leg["open_price"] or 0.0)
        * (leg["multiplier"] or 100)
        * (leg["quantity"] or 1)
        for leg in legs
    )
    d["credit_received"]      = round(credit, 2)
    d["take_profit_dollars"]  = (
        round(credit * params["take_profit_pct"], 2)
        if params.get("take_profit_pct") else None
    )
    d["stop_loss_dollars"]    = (
        -abs(params["stop_loss"])
        if params.get("stop_loss") else None
    )

    return JSONResponse(d)
