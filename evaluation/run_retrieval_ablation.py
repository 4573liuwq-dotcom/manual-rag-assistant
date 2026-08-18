"""在固定产品过滤条件下运行四阶段检索消融实验。"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / "knowledge" / ".env")
os.environ.setdefault("CHUNKS_COLLECTION", "appliance_chunks_v2")
os.environ.setdefault("ITEM_NAME_COLLECTION", "appliance_items_v2")
os.environ.setdefault("ENTITY_NAME_COLLECTION", "appliance_entities_v2")

from knowledge.processor.query_process.nodes.hyde_search_node import HydeSearchNode  # noqa: E402
from knowledge.processor.query_process.nodes.multi_search_rrf import RrfSearchNode  # noqa: E402
from knowledge.processor.query_process.nodes.multi_search_rerank import RerankSearchNode  # noqa: E402
from knowledge.utils.bgem3_client_util import (  # noqa: E402
    generate_hybrid_embeddings,
    get_bgem3_client,
)
from knowledge.utils.milvus_client_util import (  # noqa: E402
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    get_milvus_client,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def entity_list(hits: Iterable[Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for hit in hits:
        data = hit.get("entity") or hit
        if not isinstance(data, dict):
            continue
        chunk_id = data.get("chunk_id") or hit.get("id")
        docs.append({**data, "chunk_id": str(chunk_id), "retrieval_score": hit.get("distance")})
    return docs


def item_filter(item_name: str) -> str:
    escaped = item_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'item_name == "{escaped}"'


def dense_search(client, collection: str, dense: List[float], expr: str, limit: int) -> List[Dict[str, Any]]:
    result = client.search(
        collection_name=collection,
        data=[dense],
        anns_field="dense_vector",
        filter=expr,
        limit=limit,
        search_params={"metric_type": "COSINE"},
        output_fields=["chunk_id", "content", "title", "item_name", "file_title"],
    )
    return entity_list(result[0] if result else [])


def hybrid_search(client, collection: str, dense, sparse, expr: str, limit: int) -> List[Dict[str, Any]]:
    requests = create_hybrid_search_requests(dense, sparse, expr=expr, limit=limit)
    result = execute_hybrid_search_query(
        client,
        collection,
        requests,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=limit,
        output_fields=["chunk_id", "content", "title", "item_name", "file_title"],
    )
    return entity_list(result[0] if result else [])


def rrf_merge(node: RrfSearchNode, normal: List[Dict[str, Any]], hyde: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = node.rrf_merge([(normal, 1.0), (hyde, 1.0)])
    return [{**doc, "rrf_score": score} for doc, score in merged]


def measure(relevant: List[str], docs: List[Dict[str, Any]]) -> Dict[str, float]:
    expected = set(map(str, relevant))
    ids = [str(doc.get("chunk_id")) for doc in docs]
    ranks = [index + 1 for index, chunk_id in enumerate(ids) if chunk_id in expected]
    return {
        "hit_at_3": float(any(rank <= 3 for rank in ranks)),
        "recall_at_5": len({chunk_id for chunk_id in ids[:5] if chunk_id in expected}) / len(expected),
        "mrr": 1.0 / min(ranks) if ranks else 0.0,
    }


def percentile(values: List[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def summarize(rows: List[Dict[str, Any]], variants: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"case_count": len(rows), "variants": {}}
    for variant in variants:
        completed = [row[variant] for row in rows if row.get(variant, {}).get("status") == "success"]
        latencies = [row["latency_ms"] for row in completed]
        summary["variants"][variant] = {
            "success_count": len(completed),
            "hit_at_3": round(statistics.fmean(row["hit_at_3"] for row in completed), 4) if completed else 0,
            "recall_at_5": round(statistics.fmean(row["recall_at_5"] for row in completed), 4) if completed else 0,
            "mrr": round(statistics.fmean(row["mrr"] for row in completed), 4) if completed else 0,
            "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0,
            "p95_latency_ms": round(percentile(latencies, 0.95), 2),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-hyde", action="store_true")
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    collection = os.environ["CHUNKS_COLLECTION"]
    client = get_milvus_client()
    embedder = get_bgem3_client()
    hyde_node = HydeSearchNode()
    rrf_node = RrfSearchNode()
    rerank_node = RerankSearchNode()
    variants = ["dense", "hybrid"]
    if not args.skip_hyde:
        variants += ["hybrid_hyde_rrf", "hybrid_hyde_rrf_rerank"]

    rows: List[Dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            question = case["question"]
            expression = item_filter(case["item_name"])
            embedding_started = time.perf_counter()
            vectors = generate_hybrid_embeddings(embedder, [question])
            embedding_ms = (time.perf_counter() - embedding_started) * 1000
            row: Dict[str, Any] = {**case, "embedding_latency_ms": round(embedding_ms, 2)}
            try:
                started = time.perf_counter()
                dense_docs = dense_search(client, collection, vectors["dense"][0], expression, 10)
                metrics = measure(case["relevant_chunk_ids"], dense_docs)
                row["dense"] = {"status": "success", "latency_ms": round((time.perf_counter()-started)*1000 + embedding_ms, 2), "retrieved_ids": [d["chunk_id"] for d in dense_docs], **metrics}

                started = time.perf_counter()
                normal_docs = hybrid_search(client, collection, vectors["dense"][0], vectors["sparse"][0], expression, 10)
                metrics = measure(case["relevant_chunk_ids"], normal_docs)
                row["hybrid"] = {"status": "success", "latency_ms": round((time.perf_counter()-started)*1000 + embedding_ms, 2), "retrieved_ids": [d["chunk_id"] for d in normal_docs], **metrics}

                if not args.skip_hyde:
                    started = time.perf_counter()
                    hypothetical = hyde_node.generate_call_llm(question, [case["item_name"]])
                    hyde_vectors = generate_hybrid_embeddings(embedder, [f"{question}\n{hypothetical}"])
                    hyde_docs = hybrid_search(client, collection, hyde_vectors["dense"][0], hyde_vectors["sparse"][0], expression, 10)
                    rrf_docs = rrf_merge(rrf_node, normal_docs, hyde_docs)
                    rrf_latency = (time.perf_counter() - started) * 1000 + row["hybrid"]["latency_ms"]
                    metrics = measure(case["relevant_chunk_ids"], rrf_docs)
                    row["hybrid_hyde_rrf"] = {"status": "success", "latency_ms": round(rrf_latency, 2), "retrieved_ids": [d["chunk_id"] for d in rrf_docs], "hyde_document": hypothetical, **metrics}

                    started = time.perf_counter()
                    reranked = rerank_node.rerank_merged_doc(question, rrf_docs)
                    rerank_latency = rrf_latency + (time.perf_counter() - started) * 1000
                    metrics = measure(case["relevant_chunk_ids"], reranked)
                    row["hybrid_hyde_rrf_rerank"] = {"status": "success", "latency_ms": round(rerank_latency, 2), "retrieved_ids": [d["chunk_id"] for d in reranked], **metrics}
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                for variant in variants:
                    row.setdefault(variant, {"status": "error", "error": row["error"]})
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index}/{len(cases)}] {case['case_id']} 完成")

    report = summarize(rows, variants)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
