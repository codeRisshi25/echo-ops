"""
ingestion/embedder.py

Wraps fastembed (ONNX Runtime based) with an LRU cache keyed on the log *template*.

Why fastembed instead of sentence-transformers?
  sentence-transformers requires PyTorch (~2GB download).
  fastembed uses ONNX Runtime instead — same model quality, ~100MB total install.

Why cache on templates?
  Logs like "User 1234 checkout failed" and "User 5678 checkout failed"
  share the same template "User {user_id} checkout failed" → same embedding.
  In practice this gives ~80-90% cache hit rate, drastically reducing
  model inference calls and keeping Endee write volume low.
"""

import functools
from fastembed import TextEmbedding
import config

# Load once at import time. BAAI/bge-small-en-v1.5:
#   - 384 dimensions (matches ENDEE_INDEX_DIM)
#   - ~130MB, fast on CPU, free, no GPU required, ONNX-based (no PyTorch)
_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


@functools.lru_cache(maxsize=config.EMBED_CACHE_SIZE)
def _embed_template(template: str) -> tuple:
    """
    Embed a single template string. Cached — repeated templates cost nothing.
    Returns tuple (not list) so lru_cache can hash it.
    """
    # fastembed.embed() returns a generator of numpy arrays
    vector = list(_model.embed([template]))[0]
    return tuple(vector.tolist())


def get_embedding(template: str) -> list[float]:
    """
    Public API: return a 384-dim embedding for a log template.
    Result is pulled from LRU cache if template was seen before.
    """
    return list(_embed_template(template))


def get_cache_stats() -> dict:
    """Returns LRU cache efficiency stats for dashboard display."""
    info = _embed_template.cache_info()
    total = info.hits + info.misses
    hit_rate = round(info.hits / total * 100, 1) if total > 0 else 0.0
    return {
        "hits": info.hits,
        "misses": info.misses,
        "total": total,
        "hit_rate_pct": hit_rate,
        "cached_templates": info.currsize,
    }
