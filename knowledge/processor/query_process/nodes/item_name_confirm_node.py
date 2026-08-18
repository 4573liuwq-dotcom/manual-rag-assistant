import json
import re
from typing import Dict, Any, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client, create_hybrid_search_requests, \
    execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


# 类：操作llm
class ItemNameLLM():

    # 根据用户输入原始问题 + 上下文会话记录 提取商品名
    def extract_item_name(self,original_query:str,
                          history_text: str,
                          )->Dict[str,Any]:

        # 获取llm连接对象
        llm_client = get_llm_client(response_format=True)

        # 构建提示词
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text,query=original_query)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        # 调用llm llm_response是llm返回原始内容
        llm_response = llm_client.invoke(messages)
        # llm_content具体得到数据
        llm_content = llm_response.content.strip()

        #对llm_content清洗
        parse_result = self.clean_parse(llm_content)

        return parse_result

    # 清洗llm内容
    def clean_parse(self,llm_content) -> Dict[str,Any]:
        # ```json   ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 反序列
        parsed_llm_result:Dict[str,Any] = json.loads(content)

        #去掉parsed_llm_result 的item_names空格
        ori_item_names = parsed_llm_result.get('item_names') or []
        cleaned_item_names = [
            ori_item.strip()
            for ori_item in ori_item_names
            if isinstance(ori_item, str) and ori_item.strip()
        ]

        return {
            'item_names': cleaned_item_names,
            'rewritten_query': parsed_llm_result.get('rewritten_query')
        }

# 类：操作向量数据库
class ItemNameVector():
    GENERIC_ITEM_NAMES = {
        "产品", "设备", "机器", "家电", "吸尘器", "智能吸尘器", "真空吸尘器",
        "充电式真空吸尘器",
    }

    def __init__(self, config):
        self.config = config

    #根据商品名查向量数据库
    # 三件事情
    # * 1 根据item_name查询向量数据库，得到密集和稀疏向量数据
    # * 2 对查询密集和稀疏向量数据 ，评分对齐
    # 可选 3 分数差异过滤（分数大于0.7数据有多个）
    ## 返回两个列表
    ## 第一个列表：分数阈值大于0.7数据
    ## 第二个列表：分数大于0.6  小于0.7 数据
    def match_item_name_filter(self,
           item_names:List[str])->Tuple[List[str],List[str]]:
        # 1 根据item_name查询向量数据库，得到密集和稀疏向量数据
        # [{"",""} , {"",""}]
        search_result:List[Dict[str,Any]] = self.match_vector(item_names)

        # 2 根据向量数据库查询返回密集和稀疏向量，进行评分对齐
        confirmed,options = self.item_name_score_algin(search_result)
        # print("=="*50)
        # print(confirmed)
        # print("==" * 50)
        # print(options)

        # 可选 3 分数差异过滤（分数大于0.7数据有多个）
        # if len(confirmed)>1:
        #     confirmed = self.item_name_score_filter(confirmed,search_result)
        return confirmed,options

    # 根据item_name查询密集和稀疏向量
    def match_vector(self,item_names:List[str]) -> List[Dict[str,Any]]:
        # 定义列表，封装最终结果
        search_result = []

        # 获取milvus连接对象
        milvus_client = get_milvus_client()

        # 把item_names向量化查询
        embedding_model = get_bgem3_client()
        # 通过embedding_model转换密稠和稀疏向量
        # embedding_model.encode_documents(item_names)
        # 调用工具类的方法实现
        # {
        #     "dense": 密稠列表,
        #     "sparse": 稀疏向量
        # }
        hybrid_embedding_result = (
            generate_hybrid_embeddings(embedding_model,item_names))

        # item_names列表获取每个item_name 同时这个item_name对应密稠和稀疏向量
        for index,extract_name in enumerate(item_names):
            # extract_name每个商品名称
            # item_name对应密稠和稀疏向量
            dense_vector = hybrid_embedding_result['dense'][index]
            sparse_vector = hybrid_embedding_result['sparse'][index]

            # 构建混合查询条件
            # 密稠和稀疏向量，转换要求的类型 AnnSearchRequest
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
            )

            # 调研方法执行查询
            hybrid_search_result =execute_hybrid_search_query(
                milvus_client,
                collection_name=self.config.item_name_collection,
                search_requests=hybrid_search_requests,
                ranker_weights=(
                    self.config.item_name_dense_weight,
                    self.config.item_name_sparse_weight,
                ),
                norm_score=True,
                output_fields=["item_name"]
            )

            # 根据向量数据库查询结果，构建返回数据
            # 把每个item_name构建好数据放到最终列表里面 search_result
            # [
            #     [
            #         {
            #             'pk': '466823158374813291',
            #             'distance': 0.7221629619598389,
            #             'entity': {
            #                 'item_name': 'H3CLA2608室内无线网关'
            #             }
            #         }
            #     ]
            # ]
            item_name_search_result = {
                "extracted_name": extract_name,
                "matches":[
                    {
                        "item_name": h["entity"]["item_name"],
                        "score":h["distance"]
                    }
                    for h in (hybrid_search_result[0]
                              if hybrid_search_result else [])
                ]
            }
            search_result.append(item_name_search_result)
        return search_result

    # 评分对齐
    # 参数 search_result查询向量数据库返回结果 列表
    # [
    #     {
    #         "extracted_name": "H3CLA2608",
    #         "matches": [
    #             {
    #                 "item_name": "H3CLA2608",
    #                 "score": 0.7221629619598389
    #             },
    #             {
    #                 "item_name": "H3CLA2608室内无线网关1",
    #                 "score": 0.8221629619598389
    #             }
    #         ]
    #     }
    # ]
    # 返回列表元组 ，有两个列表
    # 第一个列表：分数大于0.7 数据
    # 第二个列表： 分数在0.6 -0.7之间数据
    def item_name_score_algin(self,
               search_result:List[Dict[str,Any]])->Tuple[List[str],List[str]]:
        # 定义两个列表，后面封装使用
        confirmed = [] # 分数大于0.7 数据
        options = [] # 分数在0.6 -0.7之间数据
        # 遍历参数查询结果列表
        for item_name_search_result in search_result:
            # 获取extracted_name 提取商品名称
            extracted_name = item_name_search_result.get('extracted_name')
            # 从获取名称是 matches列表，把这个列表按照score降序排列，得到排列之后列表
            # matches根据分数降序之后列表  sorted方法
            matches = sorted(item_name_search_result.get('matches'),
                   key=lambda x: x['score'],reverse=True)

            # “吸尘器”“设备”等泛称不能唯一定位型号，只能给出候选项。
            if self.is_generic_item_name(extracted_name):
                for match in matches[:self.config.item_name_max_options]:
                    if match.get("score", 0) < self.config.item_name_mid_confidence:
                        continue
                    candidate = match["item_name"]
                    if candidate not in options:
                        options.append(candidate)
                continue

            # 型号简称（如 LA2608）与库中唯一商品全称存在明确包含关系时，
            # 直接确认；若命中多个同系列商品，仍交给用户澄清。
            alias_matches = [
                match for match in matches
                if self.is_unique_model_alias(extracted_name, match.get("item_name"))
            ]
            if len(alias_matches) == 1:
                alias_item_name = alias_matches[0]["item_name"]
                if alias_item_name not in confirmed:
                    confirmed.append(alias_item_name)
                continue
            if len(alias_matches) > 1:
                for match in alias_matches[:self.config.item_name_max_options]:
                    alias_item_name = match["item_name"]
                    if alias_item_name not in options:
                        options.append(alias_item_name)
                continue

            # 1 matches列表遍历，得到列表中每部分数据分数 score值
            # 1.1 score 大于 0.7 处理 放到confirmed列表
            high = [
                m for m in matches
                if m.get('score') >= self.config.item_name_high_confidence
            ]
            if high: # 有大于评分0.7数据
                # 如果数据有一个情况
                # [{....}]
                if len(high)==1:
                    # 把值放到confirmed列表
                    high_item_name_one = high[0]["item_name"]
                    if high_item_name_one not in confirmed:
                        confirmed.append(high_item_name_one)
                # 数据有多个情况
                else: # [{....},{....}]
                    # 把值放到options列表
                    for h in high[:3]:
                        high_item_name_many = h.get("item_name")
                        if (high_item_name_many not in options
                                and high_item_name_many not in confirmed):
                            options.append(high_item_name_many)
            else: # 1.2 score 大于0.6 小于0.7 处理 放到options列表
                mid = [
                    m for m in matches
                    if m['score'] >= self.config.item_name_mid_confidence
                ]
                if mid: # 有大于0.6 小于0.7 数据
                    for m in mid[:3]:
                        m_item_name = m.get("item_name")
                        if (m_item_name not in options
                                and m_item_name not in confirmed):
                            options.append(m_item_name)
        # confirmed: 大于0.7 一个数据
        # options：1 大于0.7多个数据  2 0.6 -0.7范围之间数据
        return confirmed, options

    @staticmethod
    def is_unique_model_alias(extracted_name: str, stored_name: str) -> bool:
        """判断提取名称是否是商品全称中的明确型号简称。"""
        if not extracted_name or not stored_name:
            return False
        normalized_extracted = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]", "", extracted_name.casefold()
        )
        normalized_stored = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]", "", stored_name.casefold()
        )
        if normalized_extracted == normalized_stored:
            return True
        # 品牌可能以“松下”或“Panasonic”出现，不能把整个提取字符串做包含判断。
        # 单独抽取 8D56C、MC-8D56C、CA781 等型号 token，再与库中全称匹配。
        model_tokens = [
            re.sub(r"[^a-z0-9]", "", token)
            for token in re.findall(r"[a-z0-9-]+", extracted_name.casefold())
            if len(re.sub(r"[^a-z0-9]", "", token)) >= 4
            and any(char.isdigit() for char in token)
        ]
        return any(token in normalized_stored for token in model_tokens)

    @classmethod
    def is_generic_item_name(cls, extracted_name: str) -> bool:
        normalized = re.sub(r"[\s_-]+", "", str(extracted_name or "")).casefold()
        return normalized in cls.GENERIC_ITEM_NAMES

    # 分数差异性过滤
    # def item_name_score_filter(self):
    #     pass

# 商品名确认节点
class ItemNameConfirmNode(BaseNode):
    def __init__(self):
        super().__init__()
        # 操作llm对象
        self._item_name_llm = ItemNameLLM()
        # 操作向量数据库对象
        self._item_name_vector = ItemNameVector(self.config)

    def process(self,state:QueryGraphState)->QueryGraphState:
        # 1 获取用户输入原始问题
        original_query = state.get('original_query')

        # 2 获取用户上下文会话
        session_id = state.get('session_id')
        # 调用工具类方法，根据session_id获取最近10条会话记录
        chat_history = get_recent_messages(session_id,limit=10)
        # 把查询mongodb得到历史会话列表，拼接字符串
        history = ""
        for msg in chat_history:
            role = msg.get('role')
            text = msg.get('text')
            history += f"{role}:{text}\n"

        # 3 根据用户输入原始问题+上下文会话调用LLM 提取商品名
        # llm返回格式：在提示词约定好格式
        # {
        #     "item_names": ["商品A", "商品B"],
        #     "rewritten_query": "关于商品A和商品B，..."
        # }
        llm_result = (
            self._item_name_llm.extract_item_name(original_query,
                                                  history))
        print("调用LLM返回结果：")
        print(llm_result)
        # 4 根据LLM提取商品名，查询向量数据库进行数据处理
        # （查询得到密集和稀疏向量，评分对齐，分数差异化过滤）
        # 从上一步llm返回结果 llm_result 里面获取 商品名名称
        item_names = llm_result.get('item_names')
        rewritten_query = llm_result.get('rewritten_query')

        # 判断 item_names是否为空
        if item_names:
            # ItemNameVector类里面方法查询向量数据库
            # match_item_name_filter返回两个列表
            # 第一个列表：匹配分数阈值 0.7  大于0.7数据 []
            # 第二个列表：小于0.7数据 大于0.6数据  []
            confirmed,options = (
                self._item_name_vector.match_item_name_filter(item_names))
        else: # llm没有提取tem_name信息
            confirmed, options = [],[]

        # 5 更新state数据返回
        self.update_state(state,item_names,
                          confirmed,options,rewritten_query)

        state['history'] = chat_history
        return state

    # 更新state数据
    def update_state(self,state,item_names,
                     confirmed,options,rewritten_query):
        if confirmed: # confirmed列表：阈值分数大于0.7
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed
        elif options: #阈值分数小于0.7 大于0.6  [1,2,3] => 1,2,3
            state['answer'] = f"请选择具体问题：{','.join(options)}"
        else: #商品名没有提取出来
            state['answer'] = "当前问题无法识别..."


if __name__ == "__main__":
    test_state: QueryGraphState = {
        # "original_query": "我想知道这个如何使用？"
        "original_query": "我想知道H3C LA2608如何使用？"
    }
    test_state_json = json.dumps(test_state, ensure_ascii=False, indent=2)
    print(f"输入: {test_state_json}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result=node_item_name_confirm.process(test_state)

    print(f"确认商品: {result.get('item_names')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")
