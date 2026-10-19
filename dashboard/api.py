"""Dashboard backend: a read-only JSON API over runs/, plus the built React app.

Run:
    uv run uvicorn dashboard.api:app --port 8000
Then open http://localhost:8000 (after `npm run build` in dashboard/web), or during
frontend development run `npm run dev` there, which proxies /api to this server.
"""

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.runs import list_run_ids, run_detail, summarize

RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs"))
WEB_DIST = Path(__file__).parent / "web" / "dist"

app = FastAPI(title="Migration Agent dashboard")


@app.get("/api/runs")
def get_runs() -> list[dict[str, Any]]:
    """All recorded runs, newest first."""
    return [summarize(RUNS_DIR / run_id) for run_id in list_run_ids(RUNS_DIR)]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """One run in full. Only ids of existing run directories are accepted, so the
    path parameter can't be used to read other files (e.g. ../../.env)."""
    if run_id not in list_run_ids(RUNS_DIR):
        raise HTTPException(status_code=404, detail=f"No run {run_id!r}")
    return run_detail(RUNS_DIR / run_id)


if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """Serve the React app; client-side routes (e.g. /runs/<id>) all load index.html.
        Unknown /api/ paths stay 404s, so a broken fetch fails loudly instead of getting HTML."""
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(WEB_DIST / "index.html")
