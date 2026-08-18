"""评测 LLM 商品抽取与 Milvus 商品名确认链路。"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / "knowledge" / ".env")
os.environ.setdefault("ITEM_NAME_COLLECTION", "appliance_items_v2")

from knowledge.processor.query_process.config import get_config  # noqa: E402
from knowledge.processor.query_process.nodes.item_name_confirm_node import (  # noqa: E402
    ItemNameLLM,
    ItemNameVector,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    extractor = ItemNameLLM()
    matcher = ItemNameVector(get_config())
    rows = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            extracted = extractor.extract_item_name(case["question"], "")
            names = extracted.get("item_names") or []
            confirmed, options = matcher.match_item_name_filter(names) if names else ([], [])
            expected = set(case["expected_item_names"])
            predicted = set(confirmed)
            if case["expected_action"] == "confirm":
                correct = predicted == expected
            else:
                correct = not confirmed
            row = {
                **case,
                "status": "success",
                "extracted_item_names": names,
                "confirmed_item_names": confirmed,
                "options": options,
                "rewritten_query": extracted.get("rewritten_query", ""),
                "correct": correct,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            row = {**case, "status": "error", "correct": False, "error": f"{type(exc).__name__}: {exc}", "latency_ms": round((time.perf_counter()-started)*1000, 2)}
        rows.append(row)
        print(f"[{index}/{len(cases)}] {case['case_id']} correct={row['correct']}")
    successful = [row for row in rows if row["status"] == "success"]
    latencies = [row["latency_ms"] for row in successful]
    report = {
        "case_count": len(rows),
        "success_count": len(successful),
        "routing_accuracy": round(sum(row["correct"] for row in rows) / len(rows), 4),
        "known_item_accuracy": round(sum(row["correct"] for row in rows if row["expected_action"] == "confirm") / sum(row["expected_action"] == "confirm" for row in rows), 4),
        "unknown_rejection_rate": round(sum(row["correct"] for row in rows if row["expected_action"] != "confirm") / sum(row["expected_action"] != "confirm" for row in rows), 4),
        "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
