"""Tag management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..db import execute, query

router = APIRouter()


class TagCreate(BaseModel):
    name: str
    color: str  # hex string, e.g. '#3b82f6'


@router.get("/tags")
def list_tags() -> JSONResponse:
    cols, rows = query("SELECT id, name, color FROM tag ORDER BY name")
    return JSONResponse([dict(zip(cols, r, strict=False)) for r in rows])


@router.post("/tags", status_code=201)
def create_tag(body: TagCreate) -> JSONResponse:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Tag name must not be empty")

    existing = query("SELECT id FROM tag WHERE name = ?", [name])
    if existing[1]:
        raise HTTPException(status_code=409, detail=f"Tag '{name}' already exists")

    cols, rows = query("SELECT COALESCE(MAX(id), 0) + 1 FROM tag")
    next_id = rows[0][0]
    execute(
        "INSERT INTO tag (id, name, color) VALUES (?, ?, ?)",
        [next_id, name, body.color],
    )
    return JSONResponse({"id": next_id, "name": name, "color": body.color}, status_code=201)


@router.delete("/tags/{tag_id}", status_code=204)
def delete_tag(tag_id: int) -> JSONResponse:
    existing = query("SELECT id FROM tag WHERE id = ?", [tag_id])
    if not existing[1]:
        raise HTTPException(status_code=404, detail="Tag not found")
    execute("DELETE FROM backtest_tag WHERE tag_id = ?", [tag_id])
    execute("DELETE FROM tag WHERE id = ?", [tag_id])
    return JSONResponse(None, status_code=204)


@router.post("/backtests/{backtest_id}/tags/{tag_id}", status_code=204)
def apply_tag(backtest_id: int, tag_id: int) -> JSONResponse:
    if not query("SELECT id FROM backtest WHERE id = ?", [backtest_id])[1]:
        raise HTTPException(status_code=404, detail="Backtest not found")
    if not query("SELECT id FROM tag WHERE id = ?", [tag_id])[1]:
        raise HTTPException(status_code=404, detail="Tag not found")
    existing = query(
        "SELECT 1 FROM backtest_tag WHERE backtest_id = ? AND tag_id = ?",
        [backtest_id, tag_id],
    )
    if not existing[1]:
        execute(
            "INSERT INTO backtest_tag (backtest_id, tag_id) VALUES (?, ?)",
            [backtest_id, tag_id],
        )
    return JSONResponse(None, status_code=204)


@router.delete("/backtests/{backtest_id}/tags/{tag_id}", status_code=204)
def remove_tag(backtest_id: int, tag_id: int) -> JSONResponse:
    execute(
        "DELETE FROM backtest_tag WHERE backtest_id = ? AND tag_id = ?",
        [backtest_id, tag_id],
    )
    return JSONResponse(None, status_code=204)
