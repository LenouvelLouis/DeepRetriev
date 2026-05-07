"""Simple request metrics collector — no external dependencies.

Tracks request counts (by method, path, status) and request durations
(by method, path) in module-level dictionaries.
"""

import threading
from typing import Any

_lock = threading.Lock()

# {(method, path, status): count}
_request_counts: dict[tuple[str, str, int], int] = {}

# {(method, path): list[float]}  — durations in seconds
_request_durations: dict[tuple[str, str], list[float]] = {}


def record_request(method: str, path: str, status: int, duration: float) -> None:
    """Record a completed request.

    Args:
        method: HTTP method (GET, POST, ...).
        path: Request path.
        status: HTTP response status code.
        duration: Wall-clock duration in seconds.
    """
    with _lock:
        key_count = (method, path, status)
        _request_counts[key_count] = _request_counts.get(key_count, 0) + 1

        key_dur = (method, path)
        _request_durations.setdefault(key_dur, []).append(duration)


def get_metrics() -> dict[str, Any]:
    """Return a snapshot of collected metrics.

    Returns:
        Dict with total_requests, requests_by_endpoint,
        avg_latency_s, and request_counts_detail.
    """
    with _lock:
        total = sum(_request_counts.values())

        # Aggregate by path
        by_endpoint: dict[str, int] = {}
        for (method, path, _status), count in _request_counts.items():
            key = f"{method} {path}"
            by_endpoint[key] = by_endpoint.get(key, 0) + count

        # Average latencies by endpoint
        avg_latency: dict[str, float] = {}
        for (method, path), durations in _request_durations.items():
            key = f"{method} {path}"
            if durations:
                avg_latency[key] = round(sum(durations) / len(durations), 6)

        # Detailed counts
        detail: list[dict[str, Any]] = []
        for (method, path, status), count in _request_counts.items():
            detail.append({
                "method": method,
                "path": path,
                "status": status,
                "count": count,
            })

        return {
            "total_requests": total,
            "requests_by_endpoint": by_endpoint,
            "avg_latency_s": avg_latency,
            "request_counts_detail": detail,
        }


def reset_metrics() -> None:
    """Clear all collected metrics (useful for testing)."""
    with _lock:
        _request_counts.clear()
        _request_durations.clear()
