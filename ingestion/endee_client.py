"""
ingestion/endee_client.py

Wrapper around the official Endee Python SDK.
Using the SDK (pip install endee) is simpler and more reliable than
raw HTTP calls since Endee's API requires checksum and exact field names.

SDK docs: https://docs.endee.io
"""

from endee import Endee as _EndeeSDK, Precision
import config


class EndeeClient:
    def __init__(self, base_url: str = config.ENDEE_URL):
        self._client = _EndeeSDK()
        self._client.set_base_url(f"{base_url}/api/v1")
        self._indexes: dict = {}   # cache of index name → Index object

    # ── Index Management ──────────────────────────────────────────────────────

    def create_index(self, name: str, dim: int, metric: str = "cosine") -> str:
        """
        Create a vector index. Safe to call if index already exists.
        Uses INT16 precision — good balance of speed and accuracy.
        """
        try:
            result = self._client.create_index(
                name=name,
                dimension=dim,
                space_type=metric,
                precision="float32",   # float32 works on all hardware
            )
            return result
        except Exception as e:
            err = str(e).lower()
            if "already exists" in err or "duplicate" in err or "409" in err:
                return "already_exists"
            raise

    def _get_index(self, name: str):
        """Get or cache an Index object."""
        if name not in self._indexes:
            self._indexes[name] = self._client.get_index(name)
        return self._indexes[name]

    def list_indexes(self) -> list:
        return self._client.list_indexes()

    # ── Data Operations ───────────────────────────────────────────────────────

    def upsert(self, index_name: str, vectors: list[dict]) -> str:
        """
        Upsert a batch of vectors.
        Each item must have: id (str), vector (list[float]), metadata (dict).
        We rename 'metadata' → 'meta' to match the SDK's expected field name.
        """
        # SDK expects 'meta', our internal format uses 'metadata'
        sdk_items = [
            {
                "id": v["id"],
                "vector": v["vector"],
                "meta": v.get("metadata", v.get("meta", {})),
            }
            for v in vectors
        ]
        index = self._get_index(index_name)
        return index.upsert(sdk_items)

    def query(
        self,
        index_name: str,
        vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> dict:
        """
        Nearest-neighbour search. Returns dict with 'results' list.
        Each result has: id, score, meta.
        """
        index = self._get_index(index_name)
        raw = index.query(vector=vector, top_k=top_k, filter=filter)

        # Normalise to our internal format: {results: [{id, score, metadata}]}
        results = []
        for r in (raw or []):
            # SDK returns objects with .id, .similarity, .meta attributes
            if hasattr(r, "id"):
                results.append({
                    "id": r.id,
                    "score": getattr(r, "similarity", 0.0),
                    "metadata": getattr(r, "meta", {}),
                })
            elif isinstance(r, dict):
                results.append({
                    "id": r.get("id", ""),
                    "score": r.get("similarity", r.get("score", 0.0)),
                    "metadata": r.get("meta", r.get("metadata", {})),
                })
        return {"results": results}

    # ── Health ────────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Returns True if Endee is reachable."""
        try:
            self._client.list_indexes()
            return True
        except Exception:
            return False
