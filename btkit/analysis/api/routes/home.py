"""Home-page API: /api/studies and /api/backtests."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from ..db import cache_get, cache_set, query

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
        STDDEV_SAMP(daily_pnl) AS std_daily_pnl,
        STDDEV_SAMP(CASE WHEN daily_pnl < 0 THEN daily_pnl END) AS downside_std
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
),
equity_curve AS (
    SELECT
        p.backtest_id,
        p.exit_time,
        p.id AS pos_id,
        COALESCE(b.initial_equity, 0) + SUM(p.net_pnl) OVER (
            PARTITION BY p.backtest_id
            ORDER BY p.exit_time, p.id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS equity
    FROM position p
    JOIN backtest b ON b.id = p.backtest_id
    WHERE b.status = 'completed'
),
peak_equity AS (
    SELECT
        backtest_id,
        equity,
        MAX(equity) OVER (
            PARTITION BY backtest_id
            ORDER BY exit_time, pos_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS peak
    FROM equity_curve
),
dd_stats AS (
    SELECT backtest_id,
        MAX(peak - equity)                       AS max_drawdown,
        ARG_MAX(peak, peak - equity)             AS peak_at_max_dd
    FROM peak_equity
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
    CASE WHEN COALESCE(d.downside_std, 0.0) > 0
        THEN (d.mean_daily_pnl / d.downside_std) * SQRT(252.0)
        ELSE NULL
    END                                                              AS sortino,
    CASE WHEN COALESCE(b.initial_equity, 0) > 0
        THEN (COALESCE(t.total_pnl, 0.0) / b.initial_equity) * 100.0
        ELSE NULL
    END                                                              AS total_return_pct,
    COALESCE(dd.max_drawdown, 0.0)                                   AS max_drawdown,
    CASE WHEN COALESCE(dd.max_drawdown, 0.0) > 0
        THEN COALESCE(t.total_pnl, 0.0) / dd.max_drawdown
        ELSE NULL
    END                                                              AS recovery_factor,
    CASE WHEN COALESCE(dd.peak_at_max_dd, 0.0) > 0
        THEN (dd.max_drawdown / dd.peak_at_max_dd) * 100.0
        ELSE NULL
    END                                                              AS max_drawdown_pct,
    CASE
        WHEN t.start_date IS NOT NULL
         AND t.end_date   IS NOT NULL
         AND t.end_date > t.start_date
         AND COALESCE(b.initial_equity, 0) > 0
        THEN (
            POWER(
                (b.initial_equity + COALESCE(t.total_pnl, 0.0)) / b.initial_equity,
                365.25 / DATEDIFF('day', t.start_date, t.end_date)
            ) - 1.0
        ) * 100.0
        ELSE NULL
    END                                                              AS cagr,
    CASE
        WHEN COALESCE(dd.peak_at_max_dd, 0.0) > 0
         AND t.start_date IS NOT NULL
         AND t.end_date   IS NOT NULL
         AND t.end_date > t.start_date
         AND COALESCE(b.initial_equity, 0) > 0
        THEN (
            POWER(
                (b.initial_equity + COALESCE(t.total_pnl, 0.0)) / b.initial_equity,
                365.25 / DATEDIFF('day', t.start_date, t.end_date)
            ) - 1.0
        ) * 100.0 / (dd.max_drawdown / dd.peak_at_max_dd * 100.0)
        ELSE NULL
    END                                                              AS calmar
FROM backtest b
LEFT JOIN trade_stats t ON t.backtest_id = b.id
LEFT JOIN daily_stats  d ON d.backtest_id = b.id
LEFT JOIN dd_stats     dd ON dd.backtest_id = b.id
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

_BACKTEST_TAGS_SQL = """
SELECT bt.backtest_id, t.id, t.name, t.color
FROM backtest_tag bt
JOIN tag t ON t.id = bt.tag_id
ORDER BY bt.backtest_id, t.name
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
    return {c: _clean(v) for c, v in zip(cols, row, strict=False)}


def _extract_params(params_json: str) -> dict:
    """Pull the swept numeric params out of a strategy_params blob."""
    try:
        p = json.loads(params_json)
        result: dict[str, Any] = {}
        for trade in p.get("trades", []):
            legs = trade.get("legs", [])
            short = next(
                (leg for leg in legs if "sell" in leg.get("action", "")),
                legs[0] if legs else {},
            )
            _delta = short.get("delta") or {}
            result["delta"] = _delta.get("target") if isinstance(_delta, dict) else _delta
            result["dte"] = short.get("dte")
            ex = trade.get("exit", {})
            result["take_profit_pct"] = ex.get("take_profit_pct")
            result["stop_loss"] = ex.get("stop_loss")
            result["min_credit"] = trade.get("entry", {}).get("min_credit")
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


# ── Cache ─────────────────────────────────────────────────────────────────────

_FINGERPRINT_SQL = """
SELECT
    CAST(COUNT(*) AS VARCHAR)         || ':' ||
    CAST(COALESCE(MAX(id), 0) AS VARCHAR)  || ':' ||
    CAST(COALESCE(SUM(initial_equity), 0) AS VARCHAR) AS fp
FROM backtest
"""

_TAG_COUNT_SQL = """
SELECT CAST(COUNT(*) AS VARCHAR) FROM backtest_tag
"""


def _fingerprint() -> str:
    try:
        _, rows = query(_FINGERPRINT_SQL)
        fp = rows[0][0] if rows else "0:0:0"
    except Exception:
        fp = "0:0:0"
    try:
        _, trows = query(_TAG_COUNT_SQL)
        tc = trows[0][0] if trows else "0"
    except Exception:
        tc = "0"
    return f"{fp}:{tc}"


# ── Routes ────────────────────────────────────────────────────────────────────


def _load_backtest_tags() -> dict[int, list[dict]]:
    """Return a mapping of backtest_id → list of {id, name, color} tag dicts."""
    try:
        _, rows = query(_BACKTEST_TAGS_SQL)
    except Exception:
        return {}
    tags_by_bt: dict[int, list[dict]] = {}
    for bt_id, t_id, t_name, t_color in rows:
        tags_by_bt.setdefault(bt_id, []).append({"id": t_id, "name": t_name, "color": t_color})
    return tags_by_bt


@router.get("/backtests")
def list_backtests() -> Response:
    fp = _fingerprint()
    cached = cache_get("home.backtests", fp)
    if cached is not None:
        return Response(content=cached, media_type="application/json")

    cols, rows = query(_BACKTEST_SQL)
    tags_by_bt = _load_backtest_tags()
    result = []
    for row in rows:
        d = _row_to_dict(cols, row)
        d["params"] = _extract_params(d.pop("strategy_params"))
        d["strategy_label"] = _strategy_label(d["strategy_name"])
        d["tags"] = tags_by_bt.get(d["id"], [])
        result.append(d)
    body = json.dumps(result)
    cache_set("home.backtests", body, fp)
    return Response(content=body, media_type="application/json")


@router.get("/studies")
def list_studies() -> Response:
    fp = _fingerprint()
    cached = cache_get("home.studies", fp)
    if cached is not None:
        return Response(content=cached, media_type="application/json")

    # Fetch study basics
    s_cols, s_rows = query(_STUDY_SQL)
    studies = {r[0]: _row_to_dict(s_cols, r) for r in s_rows}

    # Fetch backtest metrics for "best" aggregation
    bt_cols, bt_rows = query(_BACKTEST_SQL)
    # Build per-study metrics
    study_metrics: dict[int, dict] = {}
    for row in bt_rows:
        d = _row_to_dict(bt_cols, row)
        sid = d["study_id"]
        sm = study_metrics.setdefault(
            sid,
            {
                "best_sharpe": None,
                "best_return_pct": None,
                "total_trades": 0,
                "data_start": None,
                "data_end": None,
                "strategies": set(),
            },
        )
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

    # Build per-study tag union from backtest tags
    tags_by_bt = _load_backtest_tags()
    # Map study_id → set of tag dicts (deduplicated by tag id)
    study_tag_map: dict[int, dict[int, dict]] = {}
    for row in bt_rows:
        d = _row_to_dict(bt_cols, row)
        sid = d["study_id"]
        if sid is None:
            continue
        for tag in tags_by_bt.get(d["id"], []):
            study_tag_map.setdefault(sid, {})[tag["id"]] = tag

    # Compose final response
    result = []
    for sid, s in studies.items():
        sm = study_metrics.get(sid, {})
        strats = sorted(sm.get("strategies") or [])
        axes = _detect_sweep_axes(study_params.get(sid, []))
        result.append(
            {
                **s,
                "strategies": strats,
                "strategy_labels": [_strategy_label(n) for n in strats],
                "sweep_axes": axes,
                "data_start": sm.get("data_start"),
                "data_end": sm.get("data_end"),
                "total_trades": sm.get("total_trades", 0),
                "best_sharpe": sm.get("best_sharpe"),
                "best_return_pct": sm.get("best_return_pct"),
                "tags": sorted(study_tag_map.get(sid, {}).values(), key=lambda t: t["name"]),
            }
        )

    body = json.dumps(result)
    cache_set("home.studies", body, fp)
    return Response(content=body, media_type="application/json")
