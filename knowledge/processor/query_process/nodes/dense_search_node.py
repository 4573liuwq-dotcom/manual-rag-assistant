"""按已确认商品过滤的纯 Dense 快速检索节点。"""

from typing import List, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bgem3_client_util import generate_hybrid_embeddings, get_bgem3_client
from knowledge.utils.milvus_client_util import get_milvus_client


class DenseSearchNode(BaseNode):
    """仅搜索 BGE-M3 稠密向量，用于默认低延迟路径。"""

    def process(self, state: QueryGraphState) -> QueryGraphState:
        item_names, rewritten_query = self.validate_param(state)
        embeddings = generate_hybrid_embeddings(get_bgem3_client(), [rewritten_query])
        if not embeddings:
            return {}
        result = get_milvus_client().search(
            collection_name=self.config.chunks_collection,
            data=[embeddings["dense"][0]],
            anns_field="dense_vector",
            filter=self.create_item_name_filter(item_names),
            limit=self.config.embedding_search_limit,
            search_params={"metric_type": "COSINE"},
            output_fields=["chunk_id", "content", "title", "file_title", "item_name"],
        )
        if not result or not result[0]:
            return {}
        return {"embedding_chunks": result[0]}

    @staticmethod
    def validate_param(state: QueryGraphState) -> Tuple[List[str], str]:
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query:
            raise ValueError("rewritten_query is empty")
        if not item_names:
            raise ValueError("item_names is empty")
        return item_names, rewritten_query

    @staticmethod
    def create_item_name_filter(item_names: List[str]) -> str:
        escaped = [name.replace("\\", "\\\\").replace('"', '\\"') for name in item_names]
        return "item_name in [" + ", ".join(f'"{name}"' for name in escaped) + "]"
