"""
In-process request metrics for GET /metrics: request volume, average
latency, and the fraud/legit split of predictions made so far.

Deliberately not Prometheus/StatsD - a single in-memory counter is
sufficient for a portfolio demo of one process, and avoids pulling in a
metrics backend for a project that doesn't have one deployed. A real
production service would export these to a proper time-series backend
instead of holding them in process memory (lost on restart, not shared
across replicas).
"""

import threading


class MetricsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._request_count = 0
        self._total_latency_ms = 0.0
        self._fraud_count = 0
        self._legit_count = 0

    def record(self, latency_ms: float, is_fraud: bool):
        with self._lock:
            self._request_count += 1
            self._total_latency_ms += latency_ms
            if is_fraud:
                self._fraud_count += 1
            else:
                self._legit_count += 1

    def summary(self) -> dict:
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._request_count if self._request_count else 0.0
            )
            return {
                "request_count": self._request_count,
                "avg_latency_ms": round(avg_latency, 2),
                "fraud_predicted_count": self._fraud_count,
                "legit_predicted_count": self._legit_count,
                "fraud_prediction_rate": round(
                    self._fraud_count / self._request_count, 4
                )
                if self._request_count
                else 0.0,
            }


metrics_tracker = MetricsTracker()
