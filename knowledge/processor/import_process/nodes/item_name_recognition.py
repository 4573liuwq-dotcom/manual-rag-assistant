import json
from typing import Tuple, List, Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.upload.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.bgem3_client_util import get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client

# 商品名识别模块
"""
    # 1 参数校验  chunks切分段数据不能为空
    # 2 获取LLM需要生成提示词数据
    # 3 构建提示词，调用LLM，提取商品名
    # 4 商品名嵌入（bge-m3生成密集和稀疏向量）
    # 5 存储到Milvus向量数据库里面
    # 6 更新state返回
"""
class ItemNameRecognition(BaseNode):
    def process(self,state:ImportGraphState)->ImportGraphState:
        # 1 参数校验  chunks切分段数据不能为空
        # 返回文件标题，切分合并列表chunks
        file_title,chunks = self.validate_param(state)
        print("第一步 参数校验。。。")
        print(file_title)

        # 2 获取LLM需要生成提示词数据
        ## 从chunks列表获取构建提取商品名提示词需要数据
        prompt_context_data = self.get_prompt_context_data(chunks)
        print("第二步 获取LLM需要生成提示词数据。。。")
        print(prompt_context_data)

        # 3 构建提示词，调用LLM，提取商品名
        item_name = self.call_llm(file_title,prompt_context_data)
        print("第三步 提取商品名。。。")
        print(item_name)

        # 4 商品名嵌入（bge-m3生成密集和稀疏向量）
        dense_vector,sparse_vector= self.embed_item_name(item_name)
        print("第四步 商品名嵌入。。。")

        # 5 存储到Milvus向量数据库里面
        self.save_to_milvus(file_title,item_name,dense_vector,sparse_vector)
        print("第五步 存储到Milvus向量数据库里面。。。")

        # 6 更新state返回
        self.update_state(item_name,state,chunks)
        return state

    # 1 参数校验方法
    def validate_param(self,state:ImportGraphState):
        # 从state获取校验参数值
        file_title = state.get('file_title')
        if not file_title:
            raise ValueError("文件标题为空")

        chunks = state.get('chunks')
        if not chunks:
            raise ValueError("chunk为空")
        return file_title,chunks

    # 2 从切片列表获取构建提示词需要数据
    # [{content:1111,file_title:qqq},{content:1333,file_title:aa}]
    # 从chunks列表获取前5切片数据，获取之后拼接字符串返回
    # 从前5个切片拼接字符串字符数量不能大于2000，如果大于2000截取
    def get_prompt_context_data(self,chunks:List[Dict[str,Any]]):
        # 定义遍历
        result = []
        total = 0
        config = get_config()
        # 遍历列表，获取前5
        for index,chunk in enumerate(chunks[:5]):
            # 从列表每段结构获取内容
            content = chunk.get('content')
            # 100
            chuck_data = f"切片-{index+1}-{content}"
            # 把每段数据放到result里面
            # 叠加判断，不超过2000
            total += len(chuck_data)
            result.append(chuck_data)
            if total > config.max_content_length:
                break
        return "\n\n".join(result)

    # 3 构建提示词，调用llm生成商品名
    def call_llm(self,file_title:str,
                 prompt_context_data:str) -> str:
        # 获取llm客户端连接对象
        llm_client = get_llm_client()
        # 构建提示词  RAFT原则
        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
            file_title=file_title,
            context=prompt_context_data
        )
        messages = [
            SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        # 调用llm
        llm_response = llm_client.invoke(messages)

        # llm_response.content
        # getattr(对象，属性，默认值)
        # setattr(llm_response,"name","zhangsan")
        item_name = getattr(llm_response,'content','')
        # 判断
        if not item_name:
            self.logger.warn("当前无法获取商品名称，返回文件名称")
            return file_title
        return item_name

    # 4 根据商品名嵌入 ，生成密集和稀疏向量
    # 参数item_name:提取商品名称
    # 返回密集向量 和 稀疏向量
    def embed_item_name(self,item_name:str):
        # 获取嵌入模型对象
        bgem3_client = get_bgem3_client()
        # 对item_name嵌入  ['商品名称','111','222']
        embedding_result = bgem3_client.encode_documents([item_name])
        # 获取embedding_result密集 和 稀疏向量
        #
        """
           encode_documents方法调用之后，返回字典
           1 有固定的key=dense，密集向量数据
               根据key=dense 获取值，值是二维数组形式 [[]] 
               传入方法文档是一个 [[]],  如果传入多个 [[],[],[]]
               
           2 有固定的key=sparse，稀疏向量数据
               稀疏向量对应稀疏矩阵，特点 token编号 对应 data权重
               返回结果不是 编号:权重，返回两个列表
               [1,2,3]   [0.1,0.3,0.6]
                      ||
               {1:0.1 , 2:0.3 , 3:0.6}
        """
        # 密集向量 dense  [[]]
        # 生成list列表
        dense = embedding_result['dense'][0].tolist()

        # 稀疏向量 sparse
        # 生成字典
        sp = embedding_result['sparse']
        # token编号
        #sp.indices.tolist()
        # data的权重
        #sp.data.tolist()
        sparse = dict(zip(sp.indices.tolist(),sp.data.tolist()))
        # 返回
        return dense,sparse

    # 5 存储到向量数据库里面
    def save_to_milvus(self,file_title,item_name,dense_vector,sparse_vector):
        #1 获取milvus连接客户端对象
        milvus_client = get_milvus_client()

        #2 判断向量数据库是否集合，如果没有创建
        config = get_config()
        # kb_item_names_v2
        # 幂等性
        if not milvus_client.has_collection(
            collection_name=config.item_name_collection):
            # 如果没有创建
            self.create_item_collection(milvus_client,config.item_name_collection)

        #3 添加数据到向量数据库里面
        data = {
            "file_title":file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }

        result = milvus_client.insert(
            collection_name=config.item_name_collection,
            data=[data]
        )
        self.logger.info(f"向量数据库添加之后结果{result}")

    # 创建集合
    def create_item_collection(self,milvus_client,collection_name):
        # （1）构建schema信息；
        schema = milvus_client.create_schema()
        # 设置主键
        schema.add_field(field_name="pk",
                         datatype=DataType.VARCHAR,
                         is_primary=True,
                         auto_id=True, # 主键生成策略，每次唯一的值
                         max_length=100)
        # 设置其他字段
        schema.add_field(field_name="item_name",
                         datatype=DataType.VARCHAR,
                         max_length=1000)
        schema.add_field(field_name="file_title",
                         datatype=DataType.VARCHAR,
                         max_length=1000)
        # 密集和稀疏向量字段
        # 1.3 稠密向量 和 稀疏向量字段约束
        schema.add_field(field_name="dense_vector",
                         datatype=DataType.FLOAT_VECTOR,
                         dim=1024)
        schema.add_field(field_name="sparse_vector",
                         datatype=DataType.SPARSE_FLOAT_VECTOR)

        # （2）添加索引；
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", # 集合字段名称
            index_name="dense_vector_index", # 索引名称
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # （3）创建collection
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    # 6 更新最终返回数据
    def update_state(self,item_name:str,state:ImportGraphState,
                     chunks:List[Dict[str,Any]]) -> None:
        for chunk in chunks:
            chunk["item_name"] = item_name
        state['item_name'] = item_name


##########################
if __name__ == '__main__':

    chunk_json_path = r"D:\dev\6W100-整本手册\auto\chunks.json"
    with open(chunk_json_path,"r",encoding="utf-8") as f:
        chunks_content = json.load(f)

    # file_title,chunks
    state = {
        "file_title":"6W100-整本手册",
        "chunks":chunks_content
    }
    itemNameRecognition = ItemNameRecognition()
    result = itemNameRecognition.process(state)

    # 把执行之后result结果输出到json文件里面
    output_path = r"D:\dev\6W100-整本手册\auto\chunks_itemname_0603.json"
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=4)


    print(result)
