from typing import List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.bgem3_client_util import generate_hybrid_embeddings, get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import create_hybrid_search_requests, execute_hybrid_search_query, \
    get_milvus_client


# hyde假设文档搜索
class HydeSearchNode(BaseNode):
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1 参数校验
        item_names,rewritten_query = self.validate_param(state)
        # 2 调用llm，根据问题生成假设为答案
        hyde_document = self.generate_call_llm(rewritten_query,item_names)
        print(f"llm返回结果：{hyde_document}")
        print("=="*50)

        # 3 用户问题 + llm返回假设为答案拼接在一起
        embedding_document = f"{rewritten_query}\n{hyde_document}"
        # 4 把拼接在一起向量化
        bgem3_client = get_bgem3_client()
        embedding_result = generate_hybrid_embeddings(bgem3_client,
                        embedding_documents=[embedding_document])

        # 5 构建查询条件：向量条件 + 标量条件
        # item_name in ["xxx", "yyy"]
        item_name_filter_expr = self.create_item_name_filter(item_names)
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filter_expr
        )

        # 6 执行混合查询
        milvus_client = get_milvus_client()
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=["chunk_id","content","item_name"]
        )
        if not res or not res[0]:
            return {}

        # 7 更新state返回
        return {"hyde_embedding_chunks":res[0]}

    # 参数校验
    def validate_param(self,state: QueryGraphState)->Tuple[List[str],str]:
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')
        if not rewritten_query:
            raise ValueError("rewritten_query is empty")
        if not item_names:
            raise ValueError("item_names is empty")
        return item_names, rewritten_query

    # 根据问题+item_name调用llm，生成假设答案
    def generate_call_llm(self,
                          rewritten_query:str,
                          item_names:List[str])->str:
        # 获取llm连接对象
        llm_client = get_llm_client()

        # 构建提示词
        system_prompt = (f"您是一位{item_names}的技术文档领域的专家，"
                         f"主要擅长编写技术文档、操作手册、文档规格说明")
        human_prompt = USER_HYDE_PROMPT_TEMPLATE.format(
            item_hint=item_names,
            rewritten_query=rewritten_query
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        # 调用
        llm_response = llm_client.invoke(messages)
        # llm_response获取具体内容
        llm_response_content = getattr(llm_response,'content',"").strip()
        if not llm_response_content:
            return ""
        return llm_response_content

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

    hyde_search_node = HydeSearchNode()
    result = hyde_search_node.process(state)
    print(result)
