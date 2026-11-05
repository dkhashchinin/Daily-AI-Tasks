"""Record per-request metrics for each LLM call into a JSONL log file."""
import json
import os
from datetime import datetime, timezone
from threading import Lock

from .config import MODEL

METRICS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics.jsonl"
)

_lock = Lock()
_last: dict | None = None


def record(
    *,
    response_model: str | None = None,
    usage: dict | None = None,
    elapsed: float | None = None,
    message_count: int | None = None,
) -> dict:
    """Build a metric payload and append it as a single JSON line."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": response_model or MODEL,
        "latency_ms": round(elapsed * 1000, 1) if elapsed is not None else None,
        "message_count": message_count,
        "usage": usage,
    }
    with _lock:
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        global _last
        _last = entry
    return entry


def last() -> dict | None:
    """Return the metrics of the most recent request, or None if none yet."""
    return _last


def describe(entry: dict | None) -> str:
    """Render one metric entry as a short human-readable line."""
    if not entry:
        return "metrics: n/a"
    usage = entry.get("usage") or {}
    p, c, t = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
    tokens = f"tokens {p}+{c}={t}" if t is not None else "tokens n/a"
    return f"metrics: {entry.get('latency_ms')} ms | {tokens} | {entry.get('model')}"


def summary() -> dict:
    """Aggregate recorded metrics into a simple summary."""
    import statistics
    from .config import MODEL

    if not os.path.exists(METRICS_FILE):
        return {"count": 0, "model": MODEL, "avg_latency_ms": None, "avg_total_tokens": None}

    entries = []
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    latencies = [e["latency_ms"] for e in entries if e.get("latency_ms") is not None]
    total_tokens = [
        e["usage"]["total_tokens"]
        for e in entries
        if e.get("usage") and e["usage"].get("total_tokens") is not None
    ]
    return {
        "count": len(entries),
        "model": entries[-1]["model"] if entries else MODEL,
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "avg_total_tokens": round(statistics.mean(total_tokens), 1) if total_tokens else None,
    }
