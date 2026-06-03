"""Home-page API: /api/studies and /api/backtests."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..db import query

router = APIRouter()

# ── SQL ───────────────────────────────────────────────────────────────────────

_BACKTEST_SQL = """
WITH
date_ranges AS (
    SELECT
        backtest_id,
        CAST(MIN(open_time) AS DATE) AS start_date,
        CAST(MAX(exit_time)  AS DATE) AS end_date
    FROM position
    GROUP BY backtest_id
),
daily_actives AS (
    SELECT
        backtest_id,
        CAST(open_time AS DATE) AS trade_date,
        SUM(net_pnl)            AS daily_pnl
    FROM position
    GROUP BY backtest_id, CAST(open_time AS DATE)
),
all_weekdays AS (
    SELECT
        dr.backtest_id,
        cal.d::DATE AS trade_date
    FROM date_ranges dr
    CROSS JOIN LATERAL (
        SELECT unnest(generate_series(dr.start_date, dr.end_date, INTERVAL '1 day')) AS d
    ) cal
    WHERE dayofweek(cal.d::DATE) NOT IN (0, 6)
),
daily AS (
    SELECT
        w.backtest_id,
        w.trade_date,
        COALESCE(a.daily_pnl, 0.0) AS daily_pnl
    FROM all_weekdays w
    LEFT JOIN daily_actives a
           ON a.backtest_id = w.backtest_id
          AND a.trade_date  = w.trade_date
),
daily_stats AS (
    SELECT backtest_id,
        AVG(daily_pnl)         AS mean_daily_pnl,
        STDDEV_SAMP(daily_pnl) AS std_daily_pnl
    FROM daily
    GROUP BY backtest_id
),
trade_stats AS (
    SELECT backtest_id,
        COUNT(*)                                                    AS n_trades,
        SUM(net_pnl)                                                AS total_pnl,
        AVG(net_pnl)                                                AS avg_pnl,
        CAST(COUNT(CASE WHEN net_pnl > 0 THEN 1 END) AS DOUBLE)
            / NULLIF(COUNT(*), 0)                                   AS win_rate,
        CAST(MIN(open_time) AS DATE)                                AS start_date,
        CAST(MAX(exit_time) AS DATE)                                AS end_date
    FROM position
    GROUP BY backtest_id
)
SELECT
    b.id,
    b.study_id,
    b.combination_id,
    b.strategy_name,
    b.strategy_params,
    b.initial_equity,
    b.status,
    b.created_at,
    COALESCE(t.n_trades, 0)                                         AS n_trades,
    COALESCE(t.total_pnl, 0.0)                                      AS total_pnl,
    t.avg_pnl,
    t.win_rate,
    t.start_date,
    t.end_date,
    CASE WHEN COALESCE(d.std_daily_pnl, 0.0) > 0
        THEN (d.mean_daily_pnl / d.std_daily_pnl) * SQRT(252.0)
        ELSE NULL
    END                                                              AS sharpe,
    CASE WHEN COALESCE(b.initial_equity, 0) > 0
        THEN (COALESCE(t.total_pnl, 0.0) / b.initial_equity) * 100.0
        ELSE NULL
    END                                                              AS total_return_pct
FROM backtest b
LEFT JOIN trade_stats t ON t.backtest_id = b.id
LEFT JOIN daily_stats  d ON d.backtest_id = b.id
WHERE b.status = 'completed'
ORDER BY b.id DESC
"""

_STUDY_SQL = """
WITH bt_counts AS (
    SELECT study_id,
        COUNT(*)                                          AS n_backtests,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) AS n_completed,
        COUNT(CASE WHEN status = 'error'     THEN 1 END) AS n_failed,
        COUNT(CASE WHEN status = 'running'   THEN 1 END) AS n_running
    FROM backtest
    GROUP BY study_id
)
SELECT
    s.id,
    s.name,
    s.note,
    s.total_combinations,
    s.created_at,
    s.finished_at,
    COALESCE(bc.n_backtests, 0) AS n_backtests,
    COALESCE(bc.n_completed,  0) AS n_completed,
    COALESCE(bc.n_failed,     0) AS n_failed,
    COALESCE(bc.n_running,    0) AS n_running
FROM study s
LEFT JOIN bt_counts bc ON bc.study_id = s.id
ORDER BY s.id DESC
"""

_STUDY_PARAMS_SQL = """
SELECT study_id, strategy_params
FROM backtest
WHERE study_id IN (SELECT id FROM study)
  AND status = 'completed'
ORDER BY study_id, combination_id
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(v: Any) -> Any:
    """Convert DB values to JSON-serialisable Python types."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def _row_to_dict(cols: list[str], row: tuple) -> dict:
    return {c: _clean(v) for c, v in zip(cols, row)}


def _extract_params(params_json: str) -> dict:
    """Pull the swept numeric params out of a strategy_params blob."""
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


def _detect_sweep_axes(params_list: list[dict]) -> list[str]:
    """Return the param keys whose values differ across combinations."""
    if not params_list:
        return []
    all_keys: set[str] = set().union(*[set(p.keys()) for p in params_list])
    return sorted(k for k in all_keys if len({str(p.get(k)) for p in params_list}) > 1)


def _strategy_label(name: str) -> str:
    """Convert snake_case strategy name to a readable label."""
    return name.replace("_", " ").title()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/backtests")
def list_backtests() -> JSONResponse:
    cols, rows = query(_BACKTEST_SQL)
    result = []
    for row in rows:
        d = _row_to_dict(cols, row)
        d["params"]         = _extract_params(d.pop("strategy_params"))
        d["strategy_label"] = _strategy_label(d["strategy_name"])
        result.append(d)
    return JSONResponse(result)


@router.get("/studies")
def list_studies() -> JSONResponse:
    # Fetch study basics
    s_cols, s_rows = query(_STUDY_SQL)
    studies = {r[0]: _row_to_dict(s_cols, r) for r in s_rows}

    # Fetch backtest metrics for "best" aggregation
    bt_cols, bt_rows = query(_BACKTEST_SQL)
    # Build per-study metrics
    study_metrics: dict[int, dict] = {}
    for row in bt_rows:
        d     = _row_to_dict(bt_cols, row)
        sid   = d["study_id"]
        sm    = study_metrics.setdefault(sid, {
            "best_sharpe":     None,
            "best_return_pct": None,
            "total_trades":    0,
            "data_start":      None,
            "data_end":        None,
            "strategies":      set(),
        })
        sm["total_trades"] += d["n_trades"]
        if d["sharpe"] is not None:
            if sm["best_sharpe"] is None or d["sharpe"] > sm["best_sharpe"]:
                sm["best_sharpe"] = round(d["sharpe"], 2)
        if d["total_return_pct"] is not None:
            if sm["best_return_pct"] is None or d["total_return_pct"] > sm["best_return_pct"]:
                sm["best_return_pct"] = round(d["total_return_pct"], 2)
        sm["strategies"].add(d["strategy_name"])
        for key, attr in [("start_date", "data_start"), ("end_date", "data_end")]:
            if d[key] is not None:
                cur = sm[attr]
                if attr == "data_start":
                    sm[attr] = d[key] if cur is None or d[key] < cur else cur
                else:
                    sm[attr] = d[key] if cur is None or d[key] > cur else cur

    # Fetch sweep axes per study
    sp_cols, sp_rows = query(_STUDY_PARAMS_SQL)
    study_params: dict[int, list[dict]] = {}
    for row in sp_rows:
        sid, params_json = row
        study_params.setdefault(sid, []).append(_extract_params(params_json))

    # Compose final response
    result = []
    for sid, s in studies.items():
        sm    = study_metrics.get(sid, {})
        strats = sorted(sm.get("strategies") or [])
        axes   = _detect_sweep_axes(study_params.get(sid, []))
        result.append({
            **s,
            "strategies":      strats,
            "strategy_labels": [_strategy_label(n) for n in strats],
            "sweep_axes":      axes,
            "data_start":      sm.get("data_start"),
            "data_end":        sm.get("data_end"),
            "total_trades":    sm.get("total_trades", 0),
            "best_sharpe":     sm.get("best_sharpe"),
            "best_return_pct": sm.get("best_return_pct"),
        })

    return JSONResponse(result)
