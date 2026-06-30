"""FastAPI application — serves both the API and the React SPA."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import home, detail, chart, compare, tags, preferences, indicators

STATIC = Path(__file__).parent.parent / "static"

app = FastAPI(title="btkit", version="2.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(home.router,   prefix="/api")
app.include_router(detail.router, prefix="/api")
app.include_router(chart.router,  prefix="/api")
app.include_router(compare.router, prefix="/api")
app.include_router(tags.router,        prefix="/api")
app.include_router(preferences.router, prefix="/api")
app.include_router(indicators.router,  prefix="/api")

# ── Static assets (hashed filenames — long-lived) ────────────────────────────
if (STATIC / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC / "assets")), name="assets")

# ── SPA fallback — any unmatched path returns index.html ─────────────────────
@app.get("/{full_path:path}", include_in_schema=False)
async def spa(full_path: str) -> FileResponse:
    candidate = STATIC / full_path
    if candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(STATIC / "index.html")
