import unittest

from evaluation.retrieval_metrics import calculate_metrics
from knowledge.processor.query_process.config import QueryConfig
from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameVector
from knowledge.processor.query_process.nodes.multi_search_rerank import RerankSearchNode
from knowledge.processor.query_process.nodes.multi_search_rrf import RrfSearchNode
from knowledge.processor.query_process.state import create_default_state


class StubAnswerOutputNode(AnswerOutputNode):
    def call_llm(self, prompt: str) -> str:
        self.received_prompt = prompt
        return "先测量设备的供电状态。[资料1]"


class AnswerOutputNodeTest(unittest.TestCase):
    def test_generates_grounded_answer_and_sources(self):
        node = StubAnswerOutputNode(config=QueryConfig(max_context_chars=500))
        state = create_default_state(
            original_query="如何排查供电问题？",
            rewritten_query="H3C LA2608 如何排查供电问题？",
            item_names=["H3C LA2608"],
            reranked_docs=[
                {
                    "chunk_id": "chunk_1",
                    "title": "供电检查",
                    "content": "先检查电源连接，再测量供电状态。",
                    "score": 0.9,
                }
            ],
        )

        result = node.process(state)

        self.assertEqual(result["answer"], "先测量设备的供电状态。[资料1]")
        self.assertEqual(result["sources"][0]["source"], "chunk_1")
        self.assertIn("[资料1]", result["prompt"])
        self.assertIn("H3C LA2608 如何排查供电问题？", result["prompt"])

    def test_refuses_when_no_evidence_exists(self):
        node = StubAnswerOutputNode(config=QueryConfig())
        state = create_default_state(original_query="一个没有资料的问题")

        result = node.process(state)

        self.assertEqual(result["answer"], node.no_evidence_answer)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["prompt"], "")

    def test_normalizes_unstable_citation_spelling(self):
        answer = "根据资料1，并参考资料2处理；资料9不在本次上下文。"
        normalized = AnswerOutputNode.normalize_citations(answer, source_count=2)
        self.assertEqual(
            normalized,
            "根据[资料1]，并[资料2]处理；资料9不在本次上下文。",
        )


class RankingNodeTest(unittest.TestCase):
    def test_model_alias_confirms_unique_full_product_name(self):
        matcher = ItemNameVector(QueryConfig())
        confirmed, options = matcher.item_name_score_algin(
            [
                {
                    "extracted_name": "LA2608",
                    "matches": [
                        {
                            "item_name": "H3C LA2608 室内无线网关",
                            "score": 0.6773,
                        }
                    ],
                }
            ]
        )

        self.assertEqual(confirmed, ["H3C LA2608 室内无线网关"])
        self.assertEqual(options, [])

    def test_brand_prefixed_short_model_alias_is_confirmed(self):
        matcher = ItemNameVector(QueryConfig())
        confirmed, options = matcher.item_name_score_algin(
            [
                {
                    "extracted_name": "松下 8D56C",
                    "matches": [
                        {
                            "item_name": "Panasonic MC-8D56C 充电式真空吸尘器",
                            "score": 0.65,
                        },
                        {"item_name": "松下 MC-8R76C 智能吸尘器", "score": 0.64},
                    ],
                }
            ]
        )
        self.assertEqual(confirmed, ["Panasonic MC-8D56C 充电式真空吸尘器"])
        self.assertEqual(options, [])

    def test_generic_product_class_requires_clarification(self):
        matcher = ItemNameVector(QueryConfig())
        confirmed, options = matcher.item_name_score_algin(
            [
                {
                    "extracted_name": "吸尘器",
                    "matches": [
                        {"item_name": "MC-CA781_MC-CA783 吸尘器", "score": 0.91},
                        {"item_name": "松下 MC-8R76C 智能吸尘器", "score": 0.82},
                    ],
                }
            ]
        )
        self.assertEqual(confirmed, [])
        self.assertEqual(len(options), 2)

    def test_rrf_accumulates_scores_for_multi_route_hit(self):
        node = RrfSearchNode(config=QueryConfig(rrf_k=60, rrf_max_results=10))
        state = create_default_state(
            embedding_chunks=[
                {"entity": {"chunk_id": "a", "content": "A"}},
                {"entity": {"chunk_id": "b", "content": "B"}},
            ],
            hyde_embedding_chunks=[
                {"entity": {"chunk_id": "b", "content": "B"}},
                {"entity": {"chunk_id": "c", "content": "C"}},
            ],
        )

        result = node.process(state)

        self.assertEqual(result["rrf_chunks"][0]["chunk_id"], "b")
        self.assertIn("rrf_score", result["rrf_chunks"][0])

    def test_cliff_cutoff_uses_configured_thresholds(self):
        node = RerankSearchNode(
            config=QueryConfig(
                rerank_max_top_k=10,
                rerank_min_top_k=3,
                rerank_gap_abs=0.5,
                rerank_gap_ratio=0.25,
            )
        )
        docs = [
            {"score": 0.95},
            {"score": 0.90},
            {"score": 0.80},
            {"score": 0.30},
            {"score": 0.20},
        ]

        self.assertEqual(len(node.cliff_cutoff(docs)), 3)


class RetrievalMetricsTest(unittest.TestCase):
    def test_calculates_hit_recall_mrr_and_item_accuracy(self):
        rows = [
            {
                "status": "success",
                "latency_ms": 100,
                "expected_chunk_ids": ["b"],
                "retrieved_ids": ["a", "b", "c"],
                "expected_item_names": ["H3C LA2608"],
                "predicted_item_names": ["H3C LA2608"],
            }
        ]

        metrics = calculate_metrics(rows, k=3)

        self.assertEqual(metrics["hit@3"], 1.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["item_name_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
