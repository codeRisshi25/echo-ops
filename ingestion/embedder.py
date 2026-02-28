"""
ingestion/embedder.py

Wraps sentence-transformers with an LRU cache keyed on the log *template* string.

Why cache on templates?
  Logs like "User 1234 purchased ITEM-456" and "User 5678 purchased ITEM-789"
  share the same template "User {user_id} purchased {item_id}" and therefore
  get the same embedding. In practice this gives ~80-90% cache hit rate,
  meaning we call the model far less and Endee gets only unique vectors.
"""

import sys
import functools
from sentence_transformers import SentenceTransformer
import config


# Load the model once at import time. Using all-MiniLM-L6-v2:
#   - 384 dimensions (matches ENDEE_INDEX_DIM)
#   - ~22MB, fast on CPU, no GPU required
#   - MIT licensed, fully free
_model = SentenceTransformer(config.EMBED_MODEL)

# Cache stats for reporting
_cache_hits = 0
_cache_misses = 0


@functools.lru_cache(maxsize=config.EMBED_CACHE_SIZE)
def _embed_template(template: str) -> tuple:
    """
    Embed a single template string. Returns a tuple (JSON-serializable)
    so we can store it in lru_cache (lists aren't hashable).
    """
    global _cache_misses
    _cache_misses += 1
    vector = _model.encode(template, normalize_embeddings=True)
    return tuple(vector.tolist())


def get_embedding(template: str) -> list[float]:
    """
    Public API: return a 384-dim embedding for a log template.
    Result is pulled from cache if the template was seen before.
    """
    global _cache_hits
    info = _embed_template.cache_info()
    vec = _embed_template(template)
    new_info = _embed_template.cache_info()
    if new_info.hits > info.hits:
        _cache_hits += 1
    return list(vec)


def get_cache_stats() -> dict:
    """Returns LRU cache efficiency stats."""
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
