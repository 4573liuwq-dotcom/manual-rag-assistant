"""Verify committed experiment evidence without model or database dependencies."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CORPUS = ROOT / "corpus" / "panasonic_vacuums"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    pdfs = sorted(CORPUS.glob("*.pdf"))
    manifest = load_jsonl(REPORTS / "panasonic_import_manifest_v2.jsonl")
    ablation = load_json(REPORTS / "panasonic_ablation_full_v1_summary.json")
    routing = load_json(REPORTS / "panasonic_item_routing_v2_summary.json")
    answers = load_json(REPORTS / "panasonic_answer_facts_v2_summary.json")
    ablation_rows = load_jsonl(REPORTS / "panasonic_ablation_full_v1.jsonl")
    routing_rows = load_jsonl(REPORTS / "panasonic_item_routing_v2.jsonl")
    answer_rows = load_jsonl(REPORTS / "panasonic_answer_facts_v2.jsonl")

    if pdfs:
        require(len(pdfs) == 4, f"若提供本地PDF，应为4份，实际{len(pdfs)}份")
    require(len(manifest) == 4, f"预期4条入库记录，实际{len(manifest)}条")
    require(all(row["status"].startswith("success") for row in manifest), "存在失败入库记录")
    require(sum(row["chunk_count"] for row in manifest) == 142, "Chunk总数不是142")
    require(sum(row["image_count"] for row in manifest) == 185, "有效图片总数不是185")
    require(ablation["case_count"] == 32 and len(ablation_rows) == 32, "检索集不是32题")
    require(routing["case_count"] == 16 and len(routing_rows) == 16, "路由集不是16题")
    require(answers["case_count"] == 8 and len(answer_rows) == 8, "答案集不是8题")

    dense = ablation["variants"]["dense"]
    enhanced = ablation["variants"]["hybrid_hyde_rrf"]
    reranked = ablation["variants"]["hybrid_hyde_rrf_rerank"]
    print("离线证据验收通过")
    corpus_note = f"本地存在{len(pdfs)}份PDF" if pdfs else "PDF不随仓库分发，入库清单已保留"
    print(f"- 语料：{corpus_note}，142个Chunk，185张有效图片")
    print(
        f"- Dense：Hit@3={dense['hit_at_3']:.2%}，MRR={dense['mrr']:.4f}，"
        f"平均={dense['avg_latency_ms'] / 1000:.2f}s"
    )
    print(
        f"- HyDE+RRF：Hit@3={enhanced['hit_at_3']:.2%}，"
        f"平均={enhanced['avg_latency_ms'] / 1000:.2f}s"
    )
    print(
        f"- +Reranker：Hit@3={reranked['hit_at_3']:.2%}，MRR={reranked['mrr']:.4f}，"
        f"平均={reranked['avg_latency_ms'] / 1000:.2f}s"
    )
    print(
        f"- 型号路由：{routing['routing_accuracy']:.2%}（{routing['case_count']}题）"
    )
    print(
        f"- 答案：关键事实召回={answers['key_fact_recall']:.2%}，"
        f"合法引用={answers['valid_citation_rate']:.2%}，"
        f"通过率={answers['answer_pass_rate']:.2%}（{answers['case_count']}题）"
    )


if __name__ == "__main__":
    main()
