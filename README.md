# Echo-Ops ⚡

> **Autonomous Infrastructure Observability Agent**  
> Detects architectural entropy in system log streams using semantic vector drift analysis — before crashes happen.

---

## The Problem

Traditional monitoring tools alert on **thresholds** — CPU > 90%, error rate > 5%. But many real failures are "silent killers":

- A database index change slows queries gradually.
- A retry storm fills up a connection pool over minutes.
- A zombie service is "up" but not processing.

These don't throw `ERROR` logs until it's too late.

## The Solution: Semantic Drift Detection

Echo-Ops treats logs as **vectors in high-dimensional space**. A healthy system has a characteristic "shape" — its logs cluster near a known-good baseline. When behaviour changes, the shape drifts — even if no explicit errors are logged.

```
Healthy Logs → Embed → Endee Baseline Index
Live Logs    → Embed → Compare vs. Baseline
                       Cosine Distance > 0.30 → ANOMALY
                                                   ↓
                                             Agentic LLM Loop
                                             calls diagnostic tools
                                                   ↓
                                             Root Cause Report
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Echo-Ops Agent                 │
│                                             │
│  ingestion/          agent/        api/     │
│  ┌───────────┐  ┌──────────────┐  ┌───────┐ │
│  │ log_gen   │  │  detector.py │  │server │ │
│  │ embedder  │→ │  (Endee ANN) │→ │  SSE  │ │
│  │  + cache  │  │  agent.py    │  │  /    │ │
│  │ endee_    │  │  (OpenRouter)│  │static │ │
│  │  client   │  │  tools.py    │  │       │ │
│  └───────────┘  └──────────────┘  └───────┘ │
└─────────────────────────────────────────────┘
         │ HTTP                   │ SSE
         ▼                        ▼
  ┌─────────────┐          ┌──────────────┐
  │  Endee DB   │          │   Browser    │
  │  :8080      │          │  Dashboard   │
  └─────────────┘          └──────────────┘
```

### How Endee Is Used

| Operation | Endee Endpoint | Purpose |
|---|---|---|
| `create_index` | `POST /api/v1/index/create` | Create `echo_ops_baseline` index (384-dim, cosine) |
| `upsert` | `POST /api/v1/index/{name}/upsert` | Batch-write healthy log embeddings (baseline) |
| `query` | `POST /api/v1/index/{name}/query` | ANN search: "how far is current log pattern from healthy?" |

**Why Endee specifically?** The drift check runs every 5 seconds against a 300+ vector index. Standard cloud vector DBs add 50–200ms of network latency per query. Endee's C++ HNSW core gives sub-millisecond local lookups, making real-time streaming analysis viable.

### Embedding Cache

The embedder caches on the **log template** (e.g. `"User {id} checkout failed"`) not the full message. Because logs repeat the same ~20 templates, we get **~80-90% cache hit rate** — drastically reducing embedding model calls and keeping Endee write volume low.

```python
@functools.lru_cache(maxsize=256)
def _embed_template(template: str) -> tuple:
    # only called on cache miss (~10-20% of logs)
    return tuple(model.encode(template).tolist())
```

### Agentic ReAct Loop

The agent is not a prompt wrapper. It's a genuine **ReAct (Reason + Act)** loop:

1. **Observe**: Receives `(service, drift_score, sample_logs)` from detector  
2. **Reason**: LLM decides which tools to call  
3. **Act**: Calls `get_recent_commits`, `get_top_db_queries`, `get_resource_snapshot`  
4. **Observe**: Tool results feed back into the conversation  
5. **Synthesize**: Produces `Root Cause Analysis Report` as structured JSON  

The LLM drives the investigation. We don't hardcode "if checkout → check DB". The agent figures that out.

---

## Project Structure

```
echo-ops/
├── config.py              # All config in one place
├── main.py                # Entry point & orchestrator
├── docker-compose.yml     # Endee vector DB
├── requirements.txt
├── ingestion/
│   ├── log_generator.py   # Synthetic log stream (healthy + anomaly)
│   ├── embedder.py        # MiniLM + LRU cache
│   └── endee_client.py    # Endee HTTP API wrapper
├── agent/
│   ├── detector.py        # Drift detection engine
│   ├── tools.py           # Diagnostic tools + OpenAI function schemas
│   └── agent.py           # Agentic LLM ReAct loop (OpenRouter)
├── api/
│   └── server.py          # FastAPI + SSE stream
└── static/
    └── index.html         # Real-time dashboard
```

---

## Setup & Run

### Prerequisites

- Python 3.11+
- Docker (for Endee)
- OpenRouter API key (free at [openrouter.ai](https://openrouter.ai))

### 1. Clone & Install

```bash
git clone https://github.com/codeRisshi25/echo-ops.git
cd echo-ops

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

### 3. Start Endee

```bash
docker compose up -d
# Verify it's running:
curl http://localhost:8080/api/v1/index/list
```

### 4. Run Echo-Ops

```bash
# Demo mode: automatically injects an anomaly at t=25s
python main.py --demo

# Open dashboard
open http://localhost:8000
```

### What you'll see

1. **t=0s** — "Building healthy baseline (300 vectors)" — Endee index populated  
2. **t=10s** — Dashboard goes live, healthy log stream visible, drift score near 0  
3. **t=25s** — Anomaly injected (checkout retry storm)  
4. **t=30s** — Drift score spikes above 0.30, agent wakes up  
5. **t=35s** — LLM calls tools (commits, DB queries, resources)  
6. **t=40s** — Root Cause Report appears in dashboard: "DB index change in checkout service"  

---

## Tech Stack

| Component | Technology | Cost |
|---|---|---|
| Vector DB | Endee (C++ HNSW) | Free, self-hosted |
| Embedding | fastembed + BAAI/bge-small-en-v1.5 (ONNX) | Free, local |
| LLM / Agent | Any OpenRouter-compatible model | Free tier available |
| API Server | FastAPI + uvicorn | Open source |
| Dashboard | Vanilla HTML/CSS/JS | — |

---

## Author

Risshi Raj Sen — [github.com/codeRisshi25](https://github.com/codeRisshi25)
