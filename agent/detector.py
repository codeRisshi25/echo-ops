"""
agent/detector.py

Drift Detection Engine — the "heartbeat" of Echo-Ops.

How it works:
1. Maintains a rolling buffer of the last DRIFT_WINDOW_SIZE log embeddings.
2. Every DRIFT_CHECK_INTERVAL_SEC seconds, computes the mean vector of the buffer.
3. Queries Endee's baseline index for the TOP_K_NEIGHBORS nearest healthy vectors.
4. Drift Score = 1 - avg(cosine_similarity of top-k results)
   → 0.0 = perfectly healthy, 1.0 = completely alien
5. If drift_score > DRIFT_THRESHOLD, emits an anomaly event.

Why the mean vector?
  A single log tells us little. 30 logs together form a "semantic fingerprint"
  of the system's current behaviour. Comparing that fingerprint to the healthy
  baseline is far more robust than per-log checks.
"""

import time
import threading
import statistics
from collections import deque

import config
from ingestion.embedder import get_embedding
from ingestion.endee_client import EndeeClient


class DriftDetector:
    def __init__(self, endee: EndeeClient, on_anomaly_callback):
        self.endee = endee
        self.on_anomaly = on_anomaly_callback       # called with (service, score, logs)
        self._buffer: deque[dict] = deque(maxlen=config.DRIFT_WINDOW_SIZE)
        self._last_drift_score = 0.0
        self._anomaly_count = 0
        self._scores_history: deque[float] = deque(maxlen=60)  # last 60 readings
        self._running = False

    def add_log(self, log: dict):
        """Feed a new log into the rolling window."""
        self._buffer.append(log)

    def get_status(self) -> dict:
        return {
            "drift_score": round(self._last_drift_score, 4),
            "anomaly_count": self._anomaly_count,
            "buffer_size": len(self._buffer),
            "is_anomaly": self._last_drift_score > config.DRIFT_THRESHOLD,
        }

    def start(self):
        """Launch the detection loop in a background thread."""
        self._running = True
        t = threading.Thread(target=self._detection_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _detection_loop(self):
        while self._running:
            time.sleep(config.DRIFT_CHECK_INTERVAL_SEC)
            if len(self._buffer) < 10:
                continue  # not enough data yet
            self._check_for_drift()

    def _check_for_drift(self):
        logs = list(self._buffer)

        # 1. Compute mean embedding of the current window
        embeddings = [get_embedding(log["template"]) for log in logs]
        dim = len(embeddings[0])
        mean_vec = [
            sum(embeddings[i][j] for i in range(len(embeddings))) / len(embeddings)
            for j in range(dim)
        ]

        # 2. Query Endee baseline for nearest healthy neighbours
        try:
            result = self.endee.query(
                index_name=config.ENDEE_INDEX_BASELINE,
                vector=mean_vec,
                top_k=config.TOP_K_NEIGHBORS,
            )
        except Exception as e:
            print(f"[Detector] Endee query error: {e}")
            return

        # 3. Endee returns scores as cosine similarity (higher = more similar)
        #    drift_score = 1 - avg_similarity
        results = result.get("results", [])
        if not results:
            return

        avg_similarity = statistics.mean(r["score"] for r in results)
        drift_score = 1.0 - avg_similarity
        
        # DEBUG
        print(f"[Detector] avg_similarity={avg_similarity:.4f} → drift_score={drift_score:.4f} (threshold={config.DRIFT_THRESHOLD})")

        self._last_drift_score = drift_score
        self._scores_history.append(drift_score)

        # 4. Fire if above threshold
        if drift_score > config.DRIFT_THRESHOLD:
            # 30-second cooldown so we don't spam the LLM
            if time.time() - getattr(self, "_last_alert_time", 0) < 30:
                pass
            else:
                self._anomaly_count += 1
                self._last_alert_time = time.time()
                # Identify the most-represented service in the buffer
                service_counts: dict[str, int] = {}
                for log in logs:
                    service_counts[log["service"]] = service_counts.get(log["service"], 0) + 1
                dominant_service = max(service_counts, key=service_counts.get)
    
                self.on_anomaly(dominant_service, drift_score, logs[-10:])
