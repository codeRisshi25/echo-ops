"""
ingestion/endee_client.py

Thin HTTP wrapper around Endee's REST API.
Endee runs at localhost:8080 (or ENDEE_URL from .env).

Endee API endpoints used:
  POST /api/v1/index/create          → create a new index
  POST /api/v1/index/{name}/upsert   → insert / update vectors
  POST /api/v1/index/{name}/query    → ANN search
  GET  /api/v1/index/list            → list all indexes
"""

import requests
import config


class EndeeClient:
    def __init__(self, base_url: str = config.ENDEE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ── Index Management ──────────────────────────────────────────────────────

    def create_index(self, name: str, dim: int, metric: str = "cosine") -> dict:
        """
        Create a vector index. Safe to call if index already exists —
        Endee returns a 409 which we silently ignore.
        """
        payload = {"name": name, "dimension": dim, "metric": metric}
        resp = self.session.post(self._url("/api/v1/index/create"), json=payload)
        if resp.status_code == 409:
            return {"status": "already_exists"}
        resp.raise_for_status()
        return resp.json()

    def list_indexes(self) -> list:
        resp = self.session.get(self._url("/api/v1/index/list"))
        resp.raise_for_status()
        return resp.json()

    # ── Data Operations ───────────────────────────────────────────────────────

    def upsert(self, index_name: str, vectors: list[dict]) -> dict:
        """
        Upsert a batch of vectors.
        Each item in `vectors` must have:
          - id:     unique string ID
          - vector: list of floats (len == dim)
          - metadata: dict of filterable fields
        """
        payload = {"vectors": vectors}
        resp = self.session.post(
            self._url(f"/api/v1/index/{index_name}/upsert"), json=payload
        )
        resp.raise_for_status()
        return resp.json()

    def query(
        self,
        index_name: str,
        vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> dict:
        """
        Nearest-neighbour search against an index.
        Returns top_k results with id, score, and metadata.

        filter example: {"service": {"$eq": "checkout"}}
        """
        payload = {"vector": vector, "top_k": top_k}
        if filter:
            payload["filter"] = filter
        resp = self.session.post(
            self._url(f"/api/v1/index/{index_name}/query"), json=payload
        )
        resp.raise_for_status()
        return resp.json()

    # ── Health ────────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Quick liveness check — returns True if Endee is up."""
        try:
            resp = self.session.get(self._url("/api/v1/index/list"), timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False
