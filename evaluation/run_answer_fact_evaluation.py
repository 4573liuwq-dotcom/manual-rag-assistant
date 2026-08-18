"""使用人工关键事实和引用规则评测最终答案，不依赖另一个裁判模型。"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / "knowledge" / ".env")
os.environ.setdefault("CHUNKS_COLLECTION", "appliance_chunks_v2")

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode  # noqa: E402
from knowledge.utils.milvus_client_util import fetch_chunks_by_chunk_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    retrieval = {
        row["case_id"]: row
        for row in (json.loads(line) for line in args.retrieval_results.read_text(encoding="utf-8").splitlines() if line)
    }
    node = AnswerOutputNode()
    rows = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        result_row = retrieval[case["case_id"]]
        ids = result_row["hybrid_hyde_rrf_rerank"]["retrieved_ids"]
        fetched = fetch_chunks_by_chunk_ids(os.environ["CHUNKS_COLLECTION"], [int(value) for value in ids])
        by_id = {str(doc["chunk_id"]): doc for doc in fetched}
        docs = [by_id[value] for value in ids if value in by_id]
        state = {
            "original_query": case["question"],
            "rewritten_query": case["question"],
            "item_names": [case["item_name"]],
            "history": [],
            "reranked_docs": docs,
        }
        try:
            result = node.process(state)
            answer = result["answer"]
            group_hits = [any(keyword in answer for keyword in group) for group in case["keyword_groups"]]
            citations = [int(value) for value in re.findall(r"\[资料(\d+)\]", answer)]
            citation_valid = bool(citations) and all(1 <= value <= len(result["sources"]) for value in citations)
            row = {
                **case,
                "status": "success",
                "answer": answer,
                "sources": result["sources"],
                "keyword_group_hits": group_hits,
                "key_fact_recall": round(sum(group_hits) / len(group_hits), 4),
                "citation_valid": citation_valid,
                "answer_pass": all(group_hits) and citation_valid,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            row = {**case, "status": "error", "answer_pass": False, "error": f"{type(exc).__name__}: {exc}", "latency_ms": round((time.perf_counter()-started)*1000, 2)}
        rows.append(row)
        print(f"[{index}/{len(cases)}] {case['case_id']} pass={row['answer_pass']}")
    successful = [row for row in rows if row["status"] == "success"]
    report = {
        "case_count": len(rows),
        "success_count": len(successful),
        "key_fact_recall": round(statistics.fmean(row["key_fact_recall"] for row in successful), 4) if successful else 0,
        "valid_citation_rate": round(statistics.fmean(float(row["citation_valid"]) for row in successful), 4) if successful else 0,
        "answer_pass_rate": round(statistics.fmean(float(row["answer_pass"]) for row in rows), 4),
        "avg_generation_latency_ms": round(statistics.fmean(row["latency_ms"] for row in successful), 2) if successful else 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
