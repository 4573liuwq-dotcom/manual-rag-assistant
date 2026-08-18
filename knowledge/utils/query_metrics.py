"""轻量级查询链路指标记录器。

默认写入 knowledge/logs/query_metrics.jsonl。每行一条 JSON，方便后续用
脚本、Excel 或可观测平台分析；记录失败不会影响主查询链路。
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


_write_lock = threading.Lock()
_LIST_FIELDS = (
    "embedding_chunks",
    "hyde_embedding_chunks",
    "web_search_docs",
    "rrf_chunks",
    "reranked_docs",
    "sources",
)


def metrics_enabled() -> bool:
    return os.getenv("QUERY_METRICS_ENABLED", "true").lower() == "true"


def metrics_path() -> Path:
    configured = os.getenv("QUERY_METRICS_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "logs" / "query_metrics.jsonl"


def summarize_state(state: Any) -> Dict[str, int]:
    if not isinstance(state, Mapping):
        return {}
    summary: Dict[str, int] = {}
    for field in _LIST_FIELDS:
        value = state.get(field)
        if isinstance(value, (list, tuple)):
            summary[field] = len(value)
    return summary


def record_node_metric(
    *,
    node_name: str,
    state: Any,
    latency_ms: float,
    status: str,
    input_counts: Dict[str, int],
    output_counts: Dict[str, int],
    error: str = "",
) -> None:
    if not metrics_enabled():
        return

    state_map: Mapping[str, Any] = state if isinstance(state, Mapping) else {}
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": str(state_map.get("task_id") or ""),
        "session_id": str(state_map.get("session_id") or ""),
        "node": node_name,
        "latency_ms": round(latency_ms, 3),
        "status": status,
        "input_counts": input_counts,
        "output_counts": output_counts,
        "error": error,
    }

    try:
        output_path = metrics_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # 可观测性不能反过来中断业务链路。
        return
