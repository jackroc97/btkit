"""Equity-curve overlay endpoints (buy & hold, live-trade CSV)."""

from __future__ import annotations

import csv as _csv
from datetime import date, datetime

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/compare/buyhold")
def compare_buyhold(
    ticker: str = Query(...),
    qty: float = Query(1.0),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> JSONResponse:
    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(400, "yfinance not installed — run: pip install yfinance") from None

    kw: dict = dict(auto_adjust=True, progress=False)
    if start:
        kw["start"] = start
    if end:
        kw["end"] = end

    data = yf.download(ticker.upper(), **kw)
    if data.empty:
        raise HTTPException(404, f"No price data found for {ticker!r}")

    close = data["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    close = close.dropna()
    if close.empty:
        raise HTTPException(404, f"No valid prices for {ticker!r}")

    first_price = float(close.iloc[0])
    dates = [d.strftime("%Y-%m-%d") for d in close.index]
    pnls = [round((float(p) - first_price) * qty, 2) for p in close.values.flatten()]

    qty_label = f" × {qty:g}" if qty != 1 else ""
    return JSONResponse(
        {
            "x": dates,
            "y": pnls,
            "name": f"{ticker.upper()} Buy & Hold{qty_label}",
        }
    )


@router.post("/compare/livetrades")
async def compare_livetrades(
    file: UploadFile = File(...),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> JSONResponse:
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    lines = text.splitlines()
    if not lines:
        raise HTTPException(400, "Empty CSV file")

    reader = _csv.DictReader(lines)
    if not reader.fieldnames:
        raise HTTPException(400, "CSV has no header row")

    fl = {h.strip().lower(): h.strip() for h in reader.fieldnames}
    date_key = next((fl[k] for k in ("date", "datetime", "time", "timestamp") if k in fl), None)
    pnl_key = next((fl[k] for k in ("pnl", "net_pnl", "profit", "realized_pnl") if k in fl), None)

    if date_key is None:
        raise HTTPException(400, "CSV must have a 'date', 'datetime', or 'timestamp' column")
    if pnl_key is None:
        raise HTTPException(400, "CSV must have a 'pnl', 'net_pnl', or 'profit' column")

    rows: list[tuple[datetime, float]] = []
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
                dt = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            continue
        if start_date and dt.date() < start_date:
            continue
        if end_date and dt.date() > end_date:
            continue
        rows.append((dt, pnl))

    if not rows:
        raise HTTPException(404, "No trades found in the specified date range")

    rows.sort(key=lambda r: r[0])
    total = 0.0
    dates: list[str] = []
    cumulative: list[float] = []
    for dt, pnl in rows:
        total += pnl
        dates.append(dt.strftime("%Y-%m-%d"))
        cumulative.append(round(total, 2))

    fname = file.filename or "CSV"
    return JSONResponse(
        {
            "x": dates,
            "y": cumulative,
            "name": f"Live Trades ({fname})",
        }
    )
