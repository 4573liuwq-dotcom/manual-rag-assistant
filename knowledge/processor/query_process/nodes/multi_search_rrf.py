from typing import List, Tuple, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

# rrf多路检索融合节点
class RrfSearchNode(BaseNode):
    def process(self, state: QueryGraphState) -> QueryGraphState:
        #1 从state获取向量检索结果 和 hyde检索结果
        vector_search_chunks = state.get('embedding_chunks') or []
        hyde_search_chunks = state.get('hyde_embedding_chunks') or []

        #2 使用rrf公式计算分数，根据分数降序
        # 2.1 构建相关数据，获取每路检索chunk文档，设置每路检索权重
        # {"vector_search_result":([{向量chunk1},{向量chun2}] , 1.0),
        #  "hyde_search_result": ([{hyde的chunk1},{hyde的chun2}] , 1.0)
        # }
        search_source = {
            "vector_search_result":
                (self.get_chunk_list(vector_search_chunks),1.0),
            "hyde_search_result":
                (self.get_chunk_list(hyde_search_chunks), 1.0)
        }

        # 为了后面计算方便
        # [([{向量chunk1},{向量chun2}] , 1.0),
        #   ([{hyde的chunk1},{hyde的chun2}] , 1.0)
        # ]
        rrf_inputs = list(search_source.values())

        # 调用方法 rrf公式计算分数，返回排序之后结果
        rrf_merge_result:List[Tuple[Dict[str,Any],float]] = (
                                self.rrf_merge(rrf_inputs))

        #3 把最终结果放到列表，更新state返回
        # [{chunk},{chunk}]
        rrf_chunks = [
            {**chunk, "rrf_score": score}
            for chunk, score in rrf_merge_result
        ]
        state['rrf_chunks'] = rrf_chunks
        return state

    # 构建查询结果数据
    # [{chunk1},{chunk2}]
    def get_chunk_list(self,rrf_inputs:List[Dict[str,Any]]
                       ) -> List[Dict[str,Any]]:
        result = []
        for doc in rrf_inputs:
            entity = doc.get('entity') or doc
            if not entity:
                continue
            result.append(entity)
        return result

    # 根据chunk和权重值，计算rrf分数，排序
    # [
    #    ([{向量chunk1},{向量chun2}] , 1.0),
    #   ([{hyde的chunk1},{hyde的chun2}] , 1.0)
    # ]
    def rrf_merge(self,rrf_inputs
           )->List[Tuple[Dict[str,Any],float]]:

        chunk_scores = {} # chunk id：分数
        chunk_data = {} # chunk id：数据
        # 遍历
        for rrf_input,weight in rrf_inputs:
            # 每部分chunks列表继续遍历
            for i,doc in enumerate(rrf_input):
                real_rank = i+1 #索引0 -》 排名1
                # chunk_id为了后面多路检索相同文档做叠加使用
                chunk_id = doc.get('chunk_id')
                if not chunk_id:
                    continue

                # rrf计算
                # 权重 / 基数 + 文档在路排名
                # {chunk_id: chunk计算分数}
                # 根据chunk id查询chunk_scores字典，是否有chunk id数据（分数）
                chunk_scores[chunk_id] = (
                        chunk_scores.get(chunk_id,float(0))
                        + weight / (self.config.rrf_k + real_rank))
                # 放chunk id ：数据
                # 相同key chunk_id只能向放一次
                chunk_data.setdefault(chunk_id,doc)

        # 按照上面计算之后每个chunk的 分数进行排序
        # 降序，分数最高前10条数据
        # [(内容,分数),(内容,分数)]
        sorted_results = sorted(
            [(chunk_data[chunk_id],score)
                for chunk_id,score in chunk_scores.items()],
            key=lambda x:x[1],reverse=True
        )
        return sorted_results[:self.config.rrf_max_results]

if __name__ == "__main__":
    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    rrf_node = RrfSearchNode()
    result = rrf_node.process(mock_state)
    print(result)
