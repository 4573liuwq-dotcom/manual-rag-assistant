"""运行端到端查询并保存逐题结果。

这个脚本会真实调用本地模型、数据库、MCP 和百炼 API。它不生成虚假指标；
某条请求失败时会记录错误并继续测试下一条。
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.processor.query_process.main_graph import query_app  # noqa: E402
from knowledge.processor.query_process.state import create_default_state  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
    return cases


def retrieved_ids(docs: Iterable[Dict[str, Any]]) -> List[str]:
    result: List[str] = []
    for doc in docs:
        identity = doc.get("chunk_id") or doc.get("url")
        if identity:
            result.append(str(identity))
    return result


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    case_id = str(case.get("case_id") or "unknown")
    question = str(case.get("question") or "").strip()
    if not question:
        raise ValueError(f"{case_id} 缺少 question")

    state = create_default_state(
        original_query=question,
        history=case.get("history") or [],
        session_id="evaluation",
        task_id=f"eval_{case_id}",
        message_id=case_id,
        is_stream=False,
    )
    started_at = time.perf_counter()
    try:
        result = query_app.invoke(state)
        latency_ms = (time.perf_counter() - started_at) * 1000
        docs = result.get("reranked_docs") or []
        return {
            **case,
            "status": "success",
            "latency_ms": round(latency_ms, 3),
            "predicted_item_names": result.get("item_names") or [],
            "rewritten_query": result.get("rewritten_query") or "",
            "retrieved_ids": retrieved_ids(docs),
            "retrieved_docs": docs,
            "answer": result.get("answer") or "",
            "sources": result.get("sources") or [],
            "error": "",
        }
    except Exception as exc:
        return {
            **case,
            "status": "error",
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "predicted_item_names": [],
            "retrieved_ids": [],
            "retrieved_docs": [],
            "answer": "",
            "sources": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行掌柜智库端到端评测")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    if args.limit > 0:
        cases = cases[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        for index, case in enumerate(cases, start=1):
            result = evaluate_case(case)
            success_count += result["status"] == "success"
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{index}/{len(cases)}] {result.get('case_id')}: "
                f"{result['status']} ({result['latency_ms']} ms)"
            )

    print(f"完成：成功 {success_count}/{len(cases)}，结果保存至 {args.output}")


if __name__ == "__main__":
    main()
