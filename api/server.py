"""
api/server.py

FastAPI server for Echo-Ops dashboard.

Endpoints:
  GET /            → serves the dashboard HTML
  GET /api/stream  → SSE stream of live log events + alerts
  GET /api/status  → current drift score and system stats
  GET /api/alerts  → all historical alerts
"""

import asyncio
import json
from pathlib import Path
from collections import deque

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

app = FastAPI(title="Echo-Ops Dashboard")

# ── Shared state (written by background threads, read by API) ─────────────────
# A thread-safe deque of SSE events waiting to be sent to clients
_event_queue: asyncio.Queue = None          # initialised on startup
_alerts: list[dict] = []                   # stored in memory (for MVP)
_status: dict = {                          # latest system status
    "drift_score": 0.0,
    "anomaly_count": 0,
    "baseline_size": 0,
    "cache_hit_rate_pct": 0.0,
    "is_anomaly": False,
}

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _event_queue
    _event_queue = asyncio.Queue(maxsize=500)


# ── Public helpers (called from background threads) ───────────────────────────

def push_log_event(log: dict):
    """Push a log line to SSE subscribers. Non-blocking."""
    if _event_queue:
        event = {"type": "log", "data": log}
        _push_event(event)

def push_alert_event(report: dict):
    """Push a root-cause alert to SSE subscribers and store it."""
    _alerts.append(report)
    if _event_queue:
        event = {"type": "alert", "data": report}
        _push_event(event)

def update_status(status: dict):
    _status.update(status)
    _push_event({"type": "status", "data": _status.copy()})

def _push_event(event: dict):
    """Try to put an event in the queue without blocking."""
    if _event_queue:
        try:
            _event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop oldest — dashboard isn't critical path


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/status")
async def get_status():
    return JSONResponse(_status)


@app.get("/api/alerts")
async def get_alerts():
    return JSONResponse(_alerts)


@app.get("/api/stream")
async def stream_events():
    """
    Server-Sent Events endpoint. The browser opens a persistent connection
    here and receives live log lines, status updates, and alerts.
    """
    async def event_generator():
        # Send initial status
        yield _sse("status", _status)
        while True:
            try:
                event = await asyncio.wait_for(_event_queue.get(), timeout=15)
                yield _sse(event["type"], event["data"])
            except asyncio.TimeoutError:
                # Send keep-alive comment to prevent connection timeout
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


def _sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
