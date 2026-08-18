import json
from typing import List, Dict, Any

from pymilvus import MilvusClient, CollectionSchema, DataType

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_client_util import get_milvus_client


# 切分向量化之后数据存储向量数据库节点
# 面向对象思想实现
# 1 内部类1: 负责构建集合的约束
class _MilvusSchemaBuilder:
    # 方法构建集合的约束
    @staticmethod
    def build(client:MilvusClient) -> CollectionSchema:
        # 创建schema对象
        # enable_dynamic_field=True
        schema = client.create_schema()
        # 设置约束：字段名称，类型，大小等
        # 主键
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        # 向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        # 标量字段
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            # nullable=True,
        )
        schema.add_field(field_name="title",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="parent_title",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="file_title",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="item_name",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="part",
                         datatype=DataType.INT64,
                         nullable=True)
        return schema

# 2 内部类2：复杂构建集合的索引
class _MilvusIndexBuilder:
    # 静态方法，构建集合索引，返回索引对象
    @staticmethod
    def build(client:MilvusClient,collection_name:str):
        index = client.prepare_index_params()
        # 为密集向量和稀疏向量创建索引
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        return index

# 3 内部类3：负责添加数据到向量数据库
# 更新数据，添加之后主键值更新数据里面
class _MilvusInsertBuilder:
    # init方法对象创建执行，传递参数
    # client：milvus连接对象
    # collection_name：集合名称
    def __init__(self, client:MilvusClient,
                     collection_name:str):
        self._client = client
        self._collection_name = collection_name

    # 添加数据库和更新数据的方法
    # 参数
    # chunks：数据列表 [{},{}..]
    # 返回结果 列表 [{},{}..]，增加字段 数据id
    def insert(self,
          chunks:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        # 添加列表数据到向量数据库
        inserted_result = self._client.insert(
            collection_name=self._collection_name,
            data=chunks,
        )
        ids = inserted_result.get('ids')
        # 更新数据的方法,把每条数据id回填到chunks里面
        self._fill_chunk_ids(chunks,ids)
        return chunks

    # 数据回填方法
    # [{数据段1},{数据段2},{数据段3}]
    # [1 ,        2,        3]
    def _fill_chunk_ids(self,chunks:List[Dict[str,Any]],ids:List[Any]):
        for chunk,id in zip(chunks,ids):
            chunk['chunk_id'] = id

# 4 节点类：主体调用流程
class ImportMilvusNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1 参数校验
        chunks = self._validate_param(state)

        # 2 获取milvus连接对象和集合名称
        milvus_client = get_milvus_client()
        config = get_config()
        collection_name = config.chunks_collection

        # 3 判断集合是否存在，不存在创建
        self.is_has_collection(milvus_client,collection_name)

        # 4 添加
        milvus_insert_obj = _MilvusInsertBuilder(client=milvus_client,
                             collection_name=collection_name)

        final_chunks = milvus_insert_obj.insert(chunks=chunks)

        # 5 更新state
        state['chunks'] = final_chunks
        return state

    # 2 判断集合是否存在
    def is_has_collection(self,
              milvus_client:MilvusClient,
                    collection_name:str):
        # 判断当前这个名称集合是否已经创建
        if not milvus_client.has_collection(collection_name):
            # 创建集合约束
            schema = _MilvusSchemaBuilder.build(milvus_client)

            # 创建集合索引
            index = _MilvusIndexBuilder.build(milvus_client,collection_name)

            # 调用方法创建
            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index,
            )

    # 1 参数校验
    def _validate_param(self,state:ImportGraphState):
        chunks = state.get('chunks')
        if not chunks:
            raise ValidationError("chunks为空")
        return chunks

if __name__ == '__main__':
    input_path = r"D:\dev\6W100-整本手册\auto\chunks_embedding_0603.json"
    with open(input_path,"r",encoding="utf-8") as f:
        file_content = json.load(f)
    state:ImportGraphState = {
        "chunks": file_content.get("chunks")
    }

    # 调用
    import_milvus = ImportMilvusNode()
    result = import_milvus.process(state)

    # 调用返回结果写入到新json文件里面
    output_path = r"D:\dev\6W100-整本手册\auto\chunks_import_milvus_0603.json"
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=4)
