import json
import re
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

from typing import Tuple, List, Any, Dict, Set

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import MilvusClient, DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.upload.import_prompt import KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from knowledge.utils.bgem3_client_util import get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client
from knowledge.utils.neo4j_util import Neo4jGraphWriter, get_neo4j_driver

# 实体名称最大长度
MAX_ENTITY_NAME_LENGTH = 15
# 实体标签白名单
ALLOWED_ENTITY_LABELS: Set[str] = {
    "Device", "Part", "Operation", "Step",
    "Warning", "Condition", "Tool",
}
# 关系类型白名单
ALLOWED_RELATION_TYPES: Set[str] = ({
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
    "MENTIONED_IN", "RELATED_TO",
})
DEFAULT_RELATION_TYPES = "RELATED_TO"

#操作Milvus数据内部类
class _MilvusEntityOperation:
    # 对象创建时候，传递milvus集合名称
    def __init__(self,collection_name:str):
        self.collection_name = collection_name

    # 删除
    # 条件：根据item_name删除
    def clear(self,milvus_client:MilvusClient,
                            item_name:str):
        collection_name = self.collection_name
        if milvus_client.has_collection(collection_name):
            milvus_client.delete(
                collection_name=collection_name,
                filter= f'item_name == "{item_name}"'
            )

    # 添加
    def insert(self,milvus_client:MilvusClient,
               entities:List[Dict],
               chunk_id:str,
               item_name:str,
               content:str):
        # entities是所有实体列表，获取列表所有实体名称
        # 遍历得到没有entity，获取每个entity里面name值非空
        # 把每个entity里面name去重操作，转换list列表
        # ['实体1','实体2']
        entities_names = list(dict.fromkeys(entity['name'] for entity in entities if entity.get('name')))

        if not entities_names:
            # raise Exception(f"实体数据为空")
            return

        # 调用方法：判断集合是否存在，如果不存在创建
        self._ensure_collection(milvus_client,self.collection_name)

        # 构建记录
        # 获取向量模型客户端对象
        bgem3_client = get_bgem3_client()
        # ['1','2']
        embedding_result = bgem3_client.encode_documents(entities_names)

        # 调用方法构建结果，添加向量数据库里面
        records = self._build_records(entities_names,embedding_result,
                            chunk_id,content,item_name)
        milvus_client.insert(
            collection_name=self.collection_name,
            data=records,
        )

    # 构建数据过程方法
    """组装插入记录。"""
    @staticmethod
    def _build_records(
            entities_names: List[str],
            embedded_result: Dict[str, Any],
            chunk_id: str,
            content: str,
            item_name: str,
    ) -> List[Dict[str, Any]]:
        """组装插入记录。"""

        # 1. 校验嵌入结果
        if not embedded_result:
            raise ValueError("嵌入结果为空")

        # 2. 获取稠密向量和稀疏向量
        dense_vector_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse")

        # 3. 校验向量是否存在
        if not dense_vector_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        # 4. 获取对应块的部分内容作为上下文
        context = content[:200]
        records: List[Dict] = []

        # 5. 遍历每一个实体名，构建记录
        for idx, entity_name in enumerate(entities_names):
            # 5.1 边界检查
            if idx >= len(dense_vector_list):
                break

            # 5.2 获取稠密向量
            dense = dense_vector_list[idx].tolist()

            # 5.3 解构稀疏向量（从 CSR 矩阵中提取当前实体的稀疏向量）
            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]

            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()

            sparse_dict = dict(zip(indices, data))

            # 5.4 构建单条记录
            record = {
                "entity_name": entity_name,
                "context": context,
                "item_name": item_name,
                "source_chunk_id": chunk_id,
                "dense_vector": dense,
                "sparse_vector": sparse_dict,
            }

            records.append(record)

        return records

    # 集合不存在则创建（schema + 索引）。
    def _ensure_collection(self, client, collection_name: str) -> None:
        """集合不存在则创建（schema + 索引）。"""

        # 1. 判断集合是否已存在
        if client.has_collection(collection_name):
            return

        # 2. 构建 schema
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=65535)
        schema.add_field("context", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)

        # 3. 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 4. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

# 知识图谱节点类
class KnowGraphNode(BaseNode):
    def __init__(self):
        super().__init__()
        config = get_config()
        self._milvus_obj = _MilvusEntityOperation(
            collection_name=config.entity_name_collection
        )
        self._neo4j_obj = Neo4jGraphWriter(
            config.neo4j_database
        )

    def process(self,state:ImportGraphState):
        self.logger.info(f"--- 开始 ---")
        # 1 参数校验
        chunks,item_name = self.validated_param(state)

        self.logger.info("知识图谱构建参数校验通过")

        # 2 删除已经存在数据，milvus 和 neo4j数据
        self.logger.info("清理当前文档的已有图谱数据")
        milvus_client = get_milvus_client()
        self._milvus_obj.clear(milvus_client,item_name)

        neo4j_driver = get_neo4j_driver()
        self._neo4j_obj.clear(neo4j_driver,item_name)

        self.logger.info("当前文档的已有图谱数据清理完成")

        # 3 数据处理
        # 把切片数据构建LLM需要提示词（最重要） **
        # 把构建提示词提交LLM，提取知识图谱数据（实体之间关系和标签） **
        # 把LLM提取出来的知识图谱数据进行清洗 **
        # 把相关数据写入到Milvus向量数据库中，同时写入到neo4j图数据库中 **
        # self.execute_all_chunks(chunks)

        # 实现多线程版本，处理知识图谱数据
        state["kg_failed_chunks"] = self.concurrent_execute_chunks(chunks)
        return state

    # # 实现多线程版本，处理知识图谱数据
    def concurrent_execute_chunks(self, chunks):
        failed_chunks = []
        with ThreadPoolExecutor(max_workers=4) as pool:

            future_list = {}
            # 遍历chunks得到每个chunk，获取数据
            for index, chunk in enumerate(chunks):
                chunk_id = chunk.get('chunk_id')
                item_name = chunk.get('item_name')
                content = chunk.get('content')

                # 向线程池里面提交任务
                # 提交之后，线程池分配线程执行任务
                future = pool.submit(
                    self.process_single_chunk,
                    chunk_id,item_name,content,
                )
                # 获取每个任务返回结果
                future_list[future] = (index, chunk_id)

            for future in as_completed(future_list):
                index,chunk_id = future_list[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(
                        "知识图谱切片处理失败 index=%s chunk_id=%s: %s",
                        index,
                        chunk_id,
                        exc,
                    )
                    failed_chunks.append(str(chunk_id))
        if failed_chunks:
            self.logger.warning(
                "知识图谱抽取完成，但有 %s 个切片未生成结构化结果",
                len(failed_chunks),
            )
        return failed_chunks

    #  3 数据处理
    ## 把切片数据构建LLM需要提示词（最重要） **
    ## 把构建提示词提交LLM，提取知识图谱数据（实体之间关系和标签） **
    # # 把LLM提取出来的知识图谱数据进行清洗 **
    ## 把相关数据写入到Milvus向量数据库中，同时写入到neo4j图数据库中 **
    def execute_all_chunks(self,chunks):
        self.logger.info("开始抽取知识图谱数据")
        # 遍历chunks
        # 对每个切片chunk，
        for i,chunk in enumerate(chunks):
            # 判断每个chunk是否字典
            if not isinstance(chunk,dict):
                continue

            # 从chunk获取数据，为了后面使用
            chunk_id = chunk.get('chunk_id')
            item_name = chunk.get('item_name')
            content = chunk.get('content')

            # 调用方法，实现
            # 构建提示词，调用llm，结果清洗，双写操作
            self.process_single_chunk(chunk_id,item_name,content)

    #  # 构建提示词，调用llm，结果清洗，双写操作
    def process_single_chunk(self,chunk_id,item_name,content):
        # 构建提示词
        messages = [
            SystemMessage(content=KNOWLEDGE_GRAPH_SYSTEM_PROMPT),
            HumanMessage(content=f"切片内容\n\n{content}"),
        ]
        # 调用llm
        llm_client = get_llm_client()
        llm_response = llm_client.invoke(messages)
        # 从content
        llm_result = getattr(llm_response,'content','').strip()
        self.logger.debug("LLM 返回知识图谱数据: %s", llm_result)

        # 把llm返回数据 结果清洗
        # graph_result 解析和清洗之后最终干净数据
        graph_result = self.parse_and_clean(llm_result)

        # graph_result获取所有实体 和 关系数据
        final_entities = graph_result.get('entities')
        final_relations = graph_result.get('relations')

        if not final_entities or not final_relations:
            return

        self.logger.debug("知识图谱数据清洗完成")

        # 双写操作 写入到milvus
        milvus_client = get_milvus_client()
        self._milvus_obj.insert(milvus_client,
            final_entities,chunk_id,content,item_name)
        self.logger.debug("实体向量写入完成")

        # 写入neo4j
        neo4j_driver = get_neo4j_driver()
        # entities, relations, chunk_id, item_name
        self._neo4j_obj.insert(neo4j_driver,final_entities,
                               final_relations,chunk_id,item_name)

    # 把llm返回数据 结果清洗
    # graph_result 解析和清洗之后最终干净数据
    def parse_and_clean(self,llm_result:str) -> Dict[str, Any]:
        # 解析llm返回数据 llm_result
        # 去掉特殊符号 （可能没有）```json      ```
        # 反序列化
        # 分别获取实体数据 和  关系数据
        # 分别对实体数据 和关系数据清洗
        # 构建最终干净实体 和 关系数据
        #1 llm_result 去掉特殊符号 （可能没有）```json     ```
        # 做法：正则表达式 + sub方法
        cleand = re.sub(r"^```(?:json)?\s*","",llm_result.strip())
        cleand = re.sub(r"\s*```$","",cleand)

        #2 cleand:str => Dict[str,Any]
        # 反序列化
        paresd_llm_response:Dict[str,Any] = json.loads(cleand)

        #3 分别获取解析之后所有实体列表 和 所有关系列表
        parsed_entities = paresd_llm_response.get('entities',[])
        parsed_relations = paresd_llm_response.get('relations', [])

        #4 调用方法，清洗实体
        # 返回清洗之后列表数据
        cleaned_entities = self.clean_entities(parsed_entities)

        #5 调用方法，清洗关系数据
        # 返回清洗之后列表数据
        # 获取清洗之后所有实体名称，关系清洗根据清洗之后实体名称进行操作
        # {key:value , key:value}
        # 没有key的字典 是 set集合： {value1, value2, value3}
        # set集合特点：不能有重复内容
        cleaned_unique_entity_name = {entity.get('name') for entity in cleaned_entities}
        cleaned_relations = self.clean_relations(cleaned_unique_entity_name,parsed_relations)

        #6 根据清洗之后返回最终实体 和 关系数据，构建最终结果
        final_data = {
            "entities":cleaned_entities,
            "relations":cleaned_relations,
        }
        return final_data

    # 清洗实体
    """
        1 清洗掉无用的实体（没有实体名称）
        2 对实体名称过长截取
        3 去掉不在实体标签白名单的标签
        4 去重
    """
    def clean_entities(self,entities:List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        # 创建set集合
        unique_data = set()

        # 放最终数据列表
        final_result = []
        # 遍历列表
        for entity in entities:
            #1 获取实体名称
            entity_name = str(entity.get('name')).strip()
            if not entity_name:
                continue

            #2 对实体名称过长截取
            # 长度不超过 15  MAX_ENTITY_NAME_LENGTH
            if len(entity_name) > MAX_ENTITY_NAME_LENGTH:
                entity_name = entity_name[:MAX_ENTITY_NAME_LENGTH]

            #3 判断标签是否在白名单里面
            entity_label = str(entity.get('label')).strip()
            if entity_label not in ALLOWED_ENTITY_LABELS:
                continue

            #4 去重
            # 规则： 实体名称+实体标签
            # 基于set集合实现去重
            unique_key = (entity_name,entity_label)
            if unique_key in unique_data:
                continue
            unique_data.add(unique_key)

            #5 构建最终返回的数据
            clean_entities = {
                "name":entity_name,
                "label":entity_label
            }

            #处理描述，如果有构建进去
            entity_description = str(entity.get('description')).strip()
            if entity_description:
                clean_entities["description"] = entity_description

            #每次循环清洗数据放到最终列表里面
            final_result.append(clean_entities)
        # 返回
        return final_result

    # 清洗关系
    """
        1 判断每个关系里面头 和 尾节点存在
        2 头和尾实体名称不能过长
        3 头和尾关系必须在关系白名单中
        4 判断头和节点是否在清洗之后所有节点集合里面
    """
    def clean_relations(self,
                        cleaned_unique_entity_name:Set[str],
                        relations:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_relations = []
        # 遍历relations得到每个关系
        for relation in relations:
            # 头和尾实体不能为空
            head_entity_name = str(relation.get('head')).strip()
            tail_entity_name = str(relation.get('tail')).strip()
            if not head_entity_name or not tail_entity_name:
                continue

            # 实体名称过长，截取
            if len(head_entity_name) > MAX_ENTITY_NAME_LENGTH:
                head_entity_name = head_entity_name[:MAX_ENTITY_NAME_LENGTH]

            if len(tail_entity_name) > MAX_ENTITY_NAME_LENGTH:
                tail_entity_name = tail_entity_name[:MAX_ENTITY_NAME_LENGTH]

            # 判断头和节点是否在清洗之后所有节点集合里面
            if (head_entity_name not in cleaned_unique_entity_name or
                    tail_entity_name not in cleaned_unique_entity_name):
                continue

            # 判断每个type关系是否在白名单里面
            relation_type = str(relation.get('type')).strip()
            if relation_type not in ALLOWED_RELATION_TYPES:
                # type添加默认关系type值
                # "RELATED_TO"
                relation_type = DEFAULT_RELATION_TYPES

            # 构建数据
            cleand_relation = {
                "head":head_entity_name,
                "tail":tail_entity_name,
                "type":relation_type
            }
            # 放到最终列表
            final_relations.append(cleand_relation)
        # 返回
        return final_relations

    # 1 参数校验
    def validated_param(self,
                    state:ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
        self.logger.info("开始知识图谱构建参数校验")

        # 1. 获取基础字段
        chunks = state.get("chunks") or []
        global_item_name = str(state.get("item_name", "")).strip()

        # 2. 校验整体 chunks 是否存在
        if not chunks:
            raise ValueError("待提取图谱的切块(chunks)不存在，跳过图谱构建。")

        # 3. 逐个校验 Chunk 的有效性
        validated_chunks = []
        for i, chunk in enumerate(chunks):

            # 3.1 chunk 是否是字典
            if not isinstance(chunk, dict):
                self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                continue

            # 3.2 处理 chunk_id
            raw_id = chunk.get("chunk_id")
            chunk_id = str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

            # 3.3 获取 content 内容
            content = str(chunk.get("content", "")).strip()
            if not content:
                self.logger.warning(f"Chunk {chunk_id} 缺少 content，已抛弃。")
                continue

            # 3.4 获取 item_name（chunk 级别优先，全局兜底）
            chunk_item = str(chunk.get("item_name", "")).strip() or global_item_name
            if not chunk_item:
                self.logger.warning(f"Chunk {chunk_id} 缺少 item_name 归属，已抛弃。")
                continue

            # 3.5 更新 chunk 字段
            chunk["chunk_id"] = chunk_id
            chunk["item_name"] = chunk_item
            chunk["content"] = content

            # 3.6 加入有效列表
            validated_chunks.append(chunk)

        # 4. 校验清洗后是否还有有效数据
        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        return validated_chunks, global_item_name

    # def _process_chunks_concurrently(self, validated_chunks:List[Dict[str,Any]], milvus_client:MilvusClient, neo4j_driver):
    #
    #     with ThreadPoolExecutor(max_workers=4) as pool:
    #         # 1. 提交所有任务
    #         future_to_idx = {}
    #         for i, chunk in enumerate(validated_chunks):
    #             content = chunk.get("content")
    #             chunk_id = str(chunk.get("chunk_id"))
    #             item_name = chunk.get("item_name")
    #
    #             # 像线程池中提交任务 返回任务对象
    #             future = pool.submit(
    #                 self._process_single_chunk,
    #                  chunk_id,item_name, content, milvus_client,neo4j_driver
    #             )
    #             future_to_idx[future] = (i, chunk_id)
    #
    #         # 2. 收集结果（按完成顺序）（一定要让执行方法的线程等所有任务做完）
    #         for future in as_completed(future_to_idx):
    #             idx, chunk_id = future_to_idx[future]
    #             try:
    #                 entity_count, relation_count = future.result()   # 任务的结果（_process_single_chunk 返回值）
    #             except Exception as e:
    #                 self.logger.error(e)

# 测试方法
def test_kg_extraction():
    """测试：模拟单个切片，跑通 LLM → 解析 → 清洗全流程。"""
    mock_state = {
        "item_name": "测试万用表",
        "chunks": [
            {
                "content": """# 电池安装
                    警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。
                    1. 把表笔与仪表断开。
                    2. 用螺丝刀拧开电池后盖上的螺母。
                    3. 正确安装电池，正负极应一致。
                    4. 盖上电池后盖并拧紧螺丝钉。
                    警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。
                    注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。""",
                "chunk_id": "18438591111",
                "item_name": "测试万用表",
            }
        ],
    }

    knowledge_graph_node = KnowGraphNode()
    knowledge_graph_node.process(mock_state)

if __name__ == "__main__":
    test_kg_extraction()
