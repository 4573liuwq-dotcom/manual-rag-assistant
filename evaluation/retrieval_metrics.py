"""从端到端评测结果计算可复现的检索和延迟指标。"""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return float(ordered[index])


def calculate_metrics(rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    latencies = [float(row.get("latency_ms") or 0) for row in successful]

    retrieval_rows = [
        row for row in successful if row.get("expected_chunk_ids")
    ]
    hits = 0
    recalls: List[float] = []
    reciprocal_ranks: List[float] = []
    for row in retrieval_rows:
        expected = {str(value) for value in row["expected_chunk_ids"]}
        retrieved = [str(value) for value in (row.get("retrieved_ids") or [])]
        top_k = retrieved[:k]
        matched = expected.intersection(top_k)
        hits += bool(matched)
        recalls.append(len(matched) / len(expected))
        first_rank = next(
            (index for index, value in enumerate(retrieved, start=1) if value in expected),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

    item_rows = [
        row for row in successful if row.get("expected_item_names")
    ]
    item_correct = 0
    for row in item_rows:
        expected_items = {str(value).casefold() for value in row["expected_item_names"]}
        predicted_items = {
            str(value).casefold() for value in (row.get("predicted_item_names") or [])
        }
        item_correct += expected_items == predicted_items

    return {
        "total_cases": len(rows),
        "successful_cases": len(successful),
        "error_cases": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "evaluated_retrieval_cases": len(retrieval_rows),
        f"hit@{k}": hits / len(retrieval_rows) if retrieval_rows else None,
        f"recall@{k}": statistics.mean(recalls) if recalls else None,
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
        "evaluated_item_cases": len(item_rows),
        "item_name_accuracy": item_correct / len(item_rows) if item_rows else None,
        "latency_avg_ms": statistics.mean(latencies) if latencies else None,
        "latency_p95_ms": percentile(latencies, 0.95) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="计算掌柜智库检索指标")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    metrics = calculate_metrics(load_jsonl(args.input), args.k)
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
