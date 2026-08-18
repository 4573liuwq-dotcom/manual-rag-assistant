import json
from typing import List, Dict, Any

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bgem3_client_util import get_bgem3_client

# 切片内容向量化节点
class ChunksEmbeddingNode(BaseNode):

    def process(self, state:ImportGraphState) -> ImportGraphState:
        # 1 参数校验 state里面chunks列表不能为空
        chunks = self.validate_param(state)

        # 2 chunks内容进行嵌入，转换密集和稀疏向量
        # 批量嵌入 每3段嵌入一次
        # 定义批量需要变量
        # 批量数量
        batch_size = 3
        # chunks大小
        total_length = len(chunks)
        # 创建列表，存储每次3段内容列表
        final_chunks = []
        # 从chunks每次取出前三段内容
        # # [1,2,3,4,5,6,7] => [1,2,3] => [4,5,6] => [7]
        # [0:3]  = 1 2 3
        # [3:6]  = 4 5 6
        # range(0,10,3)
        # for start_index in range(0, 10, 3):
        #     batch_data = chunks[0:0+3] [0,3)
        #     batch_data = chunks[3:3 + 3]
        #     batch_data = chunks[6:6 + 3]
        for start_index in range(0, total_length, batch_size):
            batch_data = chunks[start_index:start_index+batch_size]
            # 把每3段内容向量化,调用方法返回列表
            proc_result = self.exec_batch_chunks(batch_data,start_index,total_length)
            # 放到final_chunks
            # append  extend  [1,2,3]
            final_chunks.extend(proc_result)

        # 3 更新state数据
        state['chunks'] = final_chunks
        # 添加两个字段 密集向量字段和值  稀疏向量字段和值
        return state

    # 每3段切片内容向量化
    def exec_batch_chunks(self,
              batch_data:List[Dict[str,Any]],
                          start_index:int,
                          total_length:int):
        # 实现逻辑：batch_data列表遍历，得到每段{}内容，
       # 从每段内容获取content数据，把content数据调用嵌入模型向量化，
       # 返回密集和稀疏向量  ["商品名+content","商品名+content1"]
        final_contents = []
        for index,chunk  in enumerate(batch_data):
            # 从每段内容获取content数据
            content = chunk.get('content')
            item_name = chunk.get('item_name')
            # 规则：item_name + content 构成嵌入内容
            embed_content = f"{item_name}\n{content}"
            final_contents.append(embed_content)

        # 对final_contents列表内容嵌入
        bge_m3 = get_bgem3_client()
        # ["商品名+content","商品名+content1"]
        embedding_result = bge_m3.encode_documents(final_contents)

        # 从 嵌入模型返回 embedding_result结果获取每段内容密集 和 稀疏向量，
        # 更新到batch_data里面，batch_data每部分内容增加两个字段，
        #                         密集和稀疏向量字段
        for index,chunk in enumerate(batch_data):
            # 获取密集向量
            # [[],[],[]]
            # embedding_result['dense'][0].tolist()
            # embedding_result['dense'][1].tolist()
            # embedding_result['dense'][2].tolist()
            dense_vector = embedding_result['dense'][index].tolist()

            # 获取稀疏向量
            csr_result = embedding_result['sparse']

            # 获取csr指针，通过指针找到tokenid 和 权重索引
            ind_ptr = csr_result.indptr
            # [0:1]
            start_index_ind = ind_ptr[index]
            end_index_ind = ind_ptr[index+1]

            # 根据索引位置获取每段具体稀疏向量tokenid和权重值
            token_id = csr_result.indices[start_index_ind:end_index_ind].tolist()
            data = csr_result.data[start_index_ind:end_index_ind].tolist()
            # 变成字典
            sparse_vector =dict(zip(token_id,data))

            # 密集和稀疏向量更新chunk里面
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector
        return batch_data

    # 参数校验
    def validate_param(self,state:ImportGraphState):
        chunks = state.get('chunks')
        if not chunks:
            raise ValidationError("chunks内容为空")
        return chunks

if __name__ == '__main__':
    # 获取上一步内容，当前chunks_itemname_0603.json
    # 构建数据
    input_path = r"D:\dev\6W100-整本手册\auto\chunks_itemname_0603.json"
    with open(input_path,"r",encoding="utf-8") as f:
        total_content = json.load(f)

    state:ImportGraphState = {
        "chunks": total_content.get('chunks')
    }

    chunck_embed_node = ChunksEmbeddingNode()
    result = chunck_embed_node.process(state)

    # 把执行之后result结果输出到json文件里面
    output_path = r"D:\dev\6W100-整本手册\auto\chunks_embedding_0603.json"
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=4)

