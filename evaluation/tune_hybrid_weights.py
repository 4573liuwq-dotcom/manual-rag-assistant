"""搜索 BGE-M3 dense/sparse 融合权重，并保存可复现实验结果。"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / "knowledge" / ".env")
os.environ.setdefault("CHUNKS_COLLECTION", "appliance_chunks_v2")

from evaluation.run_retrieval_ablation import (  # noqa: E402
    entity_list,
    item_filter,
    load_jsonl,
    measure,
)
from knowledge.utils.bgem3_client_util import generate_hybrid_embeddings, get_bgem3_client  # noqa: E402
from knowledge.utils.milvus_client_util import (  # noqa: E402
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    get_milvus_client,
)


def search(client, collection: str, vectors, expression: str, weights, limit: int = 10):
    requests = create_hybrid_search_requests(
        vectors["dense"][0], vectors["sparse"][0], expr=expression, limit=limit
    )
    result = execute_hybrid_search_query(
        client,
        collection,
        requests,
        ranker_weights=weights,
        norm_score=True,
        limit=limit,
        output_fields=["chunk_id", "content", "title", "item_name", "file_title"],
    )
    return entity_list(result[0] if result else [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_jsonl(args.dataset)
    weights = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
    labels = [f"dense_{dense:.1f}_sparse_{sparse:.1f}" for dense, sparse in weights]
    client = get_milvus_client()
    embedder = get_bgem3_client()
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        vectors = generate_hybrid_embeddings(embedder, [case["question"]])
        row: Dict[str, Any] = {"case_id": case["case_id"]}
        for label, weight in zip(labels, weights):
            started = time.perf_counter()
            docs = search(
                client,
                os.environ["CHUNKS_COLLECTION"],
                vectors,
                item_filter(case["item_name"]),
                weight,
            )
            row[label] = {
                **measure(case["relevant_chunk_ids"], docs),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "retrieved_ids": [doc["chunk_id"] for doc in docs],
            }
        rows.append(row)
        print(f"[{index}/{len(cases)}] {case['case_id']}")

    report = {"case_count": len(rows), "weights": {}}
    for label in labels:
        report["weights"][label] = {
            "hit_at_3": round(statistics.fmean(row[label]["hit_at_3"] for row in rows), 4),
            "recall_at_5": round(statistics.fmean(row[label]["recall_at_5"] for row in rows), 4),
            "mrr": round(statistics.fmean(row[label]["mrr"] for row in rows), 4),
            "avg_search_latency_ms": round(statistics.fmean(row[label]["latency_ms"] for row in rows), 2),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": report, "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
