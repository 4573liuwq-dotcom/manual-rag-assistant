from typing import Tuple, List

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.milvus_client_util import get_milvus_client, create_hybrid_search_requests, \
    execute_hybrid_search_query

# 向量检索节点
class VectorSearchNode(BaseNode):
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1 参数校验
        item_names,rewritten_query = self.validate_param(state)

        # 2 对重写问题向量化
        # 获取嵌入模型对象
        bgem3_client = get_bgem3_client()
        # 获取milvus连接对象
        milvus_client = get_milvus_client()
        # rewritten_query向量化
        # {
        #     "dense": [密稠向量列表],
        #     "sparse": [稀疏向量列表]
        # }
        embeddings_result = generate_hybrid_embeddings(
            bgem3_client, embedding_documents=[rewritten_query])
        # 非空判断
        if not embeddings_result:
            return {}

        # 3 构建item_name标量字段条件
        # item_name in ["xxx", "yyy"]
        item_name_filter_expr = self.create_item_name_filter(item_names)
        # 4 构建问题向量化之后 向量条件
        hybrid_requests = create_hybrid_search_requests(
            # 密稠向量
            dense_vector=embeddings_result["dense"][0],
            # 稀疏向量
            sparse_vector=embeddings_result["sparse"][0],
            # 标量字段条件表达式
            expr=item_name_filter_expr,
            limit=5
        )

        # 5 执行混合检索（pymilvus的方法）
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_requests,
            # ranker_weights=(0.5, 0.5),
            norm_score=True,
            output_fields=["chunk_id", "content", "title", "file_title", "item_name"]
        )
        if not res or not res[0]:
            return {}
        # 6 更新state返回
        return {"embedding_chunks":res[0]}

    # 参数校验
    def validate_param(self,
                       state: QueryGraphState)->Tuple[List[str],str]:
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')
        if not rewritten_query:
            raise ValueError("rewritten_query is empty")
        if not item_names:
            raise ValueError("item_names is empty")
        return item_names, rewritten_query

    # 构建查询标量条件表达式
    def create_item_name_filter(self,
                item_names:List[str])->str:
        # item_name in ["xxx", "yyy"]
        item_name_filter = ", ".join(f'"{name}"' for name in item_names)
        return f" item_name in [{item_name_filter}]"

if __name__ == "__main__":
    state = {
        "rewritten_query":"关于H3C LA2608，如何使用？",
        "item_names":["H3C LA2608 室内无线网关"],
    }

    vector_search_node = VectorSearchNode()
    result = vector_search_node.process(state)
    print(result)

