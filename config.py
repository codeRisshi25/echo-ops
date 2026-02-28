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

# ── LLM Configuration (Groq Native API) ──────────────────────────────────────
# Groq provides lightning-fast inference for Llama models.
# Make sure GROQ_API_KEY is set in your environment or .env file.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set. LLM functionality may be limited.")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.3-70b-versatile"

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
DRIFT_THRESHOLD = 0.025          # cosine distance above this → anomaly
TOP_K_NEIGHBORS = 5             # how many baseline neighbors to compare

# ── Demo Mode ────────────────────────────────────────────────────────────────
DEMO_ANOMALY_INJECT_AFTER_SEC = 25  # inject anomaly N seconds into demo

# ── Dashboard ────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
