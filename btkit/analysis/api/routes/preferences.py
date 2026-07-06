"""UI preference persistence: GET/PUT /api/preferences/{key}."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any

from ..db import pref_get, pref_set

router = APIRouter()


class PrefBody(BaseModel):
    value: Any


@router.get("/preferences/{key:path}")
def get_preference(key: str) -> JSONResponse:
    value = pref_get(key)
    return JSONResponse({"value": value})


@router.put("/preferences/{key:path}")
def set_preference(key: str, body: PrefBody) -> JSONResponse:
    pref_set(key, body.value)
    return JSONResponse({"ok": True})
