"""
config.py — Central configuration for Echo-Ops.
All tuneable knobs in one place. No magic numbers scattered elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Endee Vector DB ──────────────────────────────────────────────────────────
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_INDEX_BASELINE = "echo_ops_baseline"
ENDEE_INDEX_DIM = 384          # all-MiniLM-L6-v2 output dimension
ENDEE_METRIC = "cosine"

# ── OpenRouter LLM ───────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Fast, free, function-calling capable
LLM_MODEL = "google/gemini-2.0-flash-exp:free"

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # local, free, 384-dim
EMBED_CACHE_SIZE = 256              # max unique templates to cache

# ── Ingestion ────────────────────────────────────────────────────────────────
BATCH_SIZE = 20          # upsert to Endee every N logs
BATCH_TIMEOUT_SEC = 3   # or every N seconds, whichever is first
BASELINE_LOG_COUNT = 300 # number of healthy logs to build the baseline

# ── Drift Detection ──────────────────────────────────────────────────────────
DRIFT_CHECK_INTERVAL_SEC = 5    # how often to check for drift
DRIFT_WINDOW_SIZE = 30          # rolling window of recent logs to analyze
DRIFT_THRESHOLD = 0.30          # cosine distance above this → anomaly
TOP_K_NEIGHBORS = 5             # how many baseline neighbors to compare

# ── Demo Mode ────────────────────────────────────────────────────────────────
DEMO_ANOMALY_INJECT_AFTER_SEC = 25  # inject anomaly N seconds into demo

# ── Dashboard ────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
