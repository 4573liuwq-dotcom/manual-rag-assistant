import json
import os.path
import re
from typing import Tuple, List, Any, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


# 文档切分节点
class DocumentSplitNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 加载 -- 切分（字符、语义、标题） -- 组合（大继续切分，小合并，不大不小保存）
        # 基于md文档，根据标题切分
        #1 获取参数，格式标准化 \r \r\n => \n
        (md_content,file_title) = self.get_input_param(state)

        #2 根据md的标题切分
        parts = self.split_doc_title(md_content,file_title)
        self.logger.info("按标题切分得到 %s 个章节", len(parts))

        #3 对根据标题切分内容再进行处理
        # [{body:111,title:第一章}，{body:666,title:第2章}]
        # 原则：如果每段内容过大再进行切分
        # 3.1 如果段内容过大切分，阈值每段最大不超过1000
        #     流程： 判断段内容如果小于1000，不切分返回
        #           判断段内容大于1000，切分，递归字符文本切分器
        #            ["\n\n","\n","。","！","？"," ",""]
        # 3.2 如果每段内容太小，多个段内容合并，阈值每段最小500
        #     流程：判断每段内容大于500，不合并返回
        #          判断每段内容小于500，进行多段合并
        #          合并规则：同源合并，找同一个parent_title段内容合并
        #     特殊：如果合并之后总大小大于100，不做处理
        config = get_config()
        final_chunks = self.split_and_merge(parts,
                    config.max_content_length,config.min_content_length)
        self.logger.info("长段拆分与短段合并后得到 %s 个切片", len(final_chunks))

        #4 对上面切分和合并返回列表，根据业务需求重新组装，组装满足具体业务结构
        chunks = self.collect_data(final_chunks)
        self.logger.info("最终组装 %s 个知识切片", len(chunks))
        # 更新到state里面
        state['chunks'] = chunks
        #5 其他：输出操作日志 和 备份json格式
        self.output_log(md_content,chunks,config.max_content_length)
        # 备份json格式
        self.copy_chunks(chunks,state)
        return state

    # 把拆分合并之后，最终组装的数据备份
    # 把数据生成json格式文件，保存到本地目录
    def copy_chunks(self,chunks,state):
        file_dir = state.get('file_dir')
        # "file_dir": r"D:\dev\test\a.json"
        output_file_dir = os.path.join(file_dir,'chunks.json')
        with open(output_file_dir,'w',encoding='utf-8') as f:
            json.dump(chunks,f,ensure_ascii=False,indent=4)

    # 输出日志
    def output_log(self,md_content,chunks,max_content_length):
        # 统计md文档行
        line_count = md_content.count("\n")+1
        self.logger.info(f"md文档共有{line_count}行")
        self.logger.info(f"最终切分章节数:{len(chunks)}")
        self.logger.info(f"最大切片长度:{max_content_length}")

        if chunks:
            self.logger.info("章节预览:")  # 只前5个章节，做简单预览；i为下标，sec为单个章节字典
            for i, sec in enumerate(chunks[:5]):
                # 取出标题，并且截断为前30个字符
                title = sec.get("title", "")[:30]
                self.logger.info(f"{i + 1}.{title}")
            if len(chunks) > 5:
                self.logger.info(f"还有{len(chunks) - 5}章节")

    # 对切分和合并数据重新组装
    def collect_data(self,final_chunks:List[Dict[str,Any]]) -> List[Dict[str,Any]]:

        chunks = []
        for chunk in final_chunks:
            # {
            #     "title": title + "-" + f"{index + 1}",
            #     "body": text,
            #     "file_title": file_title,
            #     "parent_title": parent_title,
            #     "part": f"{index + 1}",
            # }
            title = chunk.get('title')
            body = chunk.get('body')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')

            # 最终内容 ：content 包含 title + body
            content = f"{title}\n\n{body}"

            data = {
                "title": title,
                "content": content,
                "file_title": file_title,
                "parent_title": parent_title,
            }
            if "part" in chunk:
                data['part'] = chunk.get('part')
            chunks.append(data)
        return chunks

    # 把标题切分文档拆分和合并
    def split_and_merge(self, parts:List[Dict[str, Any]],
                        max_content_length:int,
                        min_content_length:int):
        # 1 对parts列表遍历，获取每段内容
        current_parts = []
        for part in parts:
            # 对每段内容，如果过大，再拆分
            # ["1","2","3"]
            # chunks_split_long = self.split_long_part(part, max_content_length)
            # current_parts.extend(chunks_split_long)
            current_parts.extend(self.split_long_part(part, max_content_length))

        # 2 把上一步拆分之后每段内容，进行合并
        final_parts = self.merge_short_part(current_parts,min_content_length)
        return final_parts

    # 内容过大继续切分
    def split_long_part(self,part,max_content_length):

        # {
        #     "title": new_title,
        #     "body": content,
        #     "file_title": file_title,
        #     "parent_title": parent_title,
        # }
        #1 从每段内容part字典获取相关数据
        title = part.get('title')
        body = part.get('body')
        file_title = part.get('file_title')
        parent_title = part.get('parent_title')

        #2 判断当前段内容是否超过最大大小 max_content_length
        # title处理
        if len(title) > 50:
            title = title[:50]

        # 计算title + body 总大小
        # 一般 切分大小计算：包含 title和body
        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)
        # total_length max_content_length比较
        # 如果这段内容比要求大小小，直接返回
        if total_length <= max_content_length:
            return [part]

        # 获取切分每段大小
        body_size = max_content_length - len(title_prefix)

        # 如果这段内容比要求大，继续切分
        # 使用langchain封装递归字符文本切分器
        text_splitter = RecursiveCharacterTextSplitter(
            # 切分每段大小是多大
            chunk_size=body_size,
            chunk_overlap=0,
            separators=["\n\n","\n","。","！","？"," ",""],
            keep_separator=False,
        )
        # 返回列表
        texts = text_splitter.split_text(body)

        # 把切分之后列表，再处理，返回业务相关字段和值
        final_parts = []
        # 递归字符文本切分器返回列表遍历
        for index,text in enumerate(texts):
            final_parts.append({
                "title": title+"-"+f"{index+1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index+1}",
            })
        return final_parts

    # 切分内容太小合并
    def merge_short_part(self,current_parts,min_content_length):
        # current_parts传递切分之后列表 [{...},{....}]
        # 获取列表第一段内容,以第一段为基准
        current_part = current_parts[0]

        # 创建列表，存最终数据
        final_parts = []

        # 从第二段内容开始遍历
        # 贪婪算法
        for next_part in current_parts[1:]:
            # 判断下一段内容和当前内容是否同源 是否同一个parent_title
            same_parent = current_part["parent_title"] == next_part["parent_title"]
            # 当前内容大小是否小于min_content_length
            body_len = len(current_part.get('body'))
            if same_parent and body_len<min_content_length:
                # 当前段内容 body  + 下一段内容 body
                #    1111  lstrip()  rstrip()  strip()
                current_part['body'] = current_part.get('body').rstrip() +"\n\n"+ next_part.get('body').lstrip()
                current_part['title'] = current_part["parent_title"]
                current_part['part'] = 0
            else:
                final_parts.append(current_part)
                current_part = next_part
        # 处理最后一段内容
        final_parts.append(current_part)

        # 专门处理每段内容part字段（根据业务也可以不处理）
        part_counter = {}
        result = []
        for parts in final_parts:
            if "part" in parts:
                parent_title = parts["parent_title"]

                part_counter[parent_title] = (
                        part_counter.get(parent_title, 0) + 1)

                new_part = part_counter[parent_title]

                parts["part"] = new_part
            result.append(parts)
        return result

    # 根据标题切分
    # 从第一行开始向下操作：遇到标题结束，上面切分一段
    # [{body:111,title:第一章}，{body:666,title:第2章}]
    def split_doc_title(self,
         md_content:str,file_title:str)->List[dict]:
        # 创建查找标题正则表达式
        title_rule = re.compile(r"^\s*(#{1,6})\s+(.+)")
        # 根据\n获取md每行内容
        lines = md_content.split("\n")

        # 列表：用于临时存储
        temp = []
        # 列表：最终数据
        res = []
        # 记录标题
        new_title = ""
        # 是否代码块
        in_code = False

        # 添加列表，用于存储每个级别标题
        # 一级标题 # 内容  ， 二级标题 ## 内容
        #    0   1    2     3
        #  ["","# ","## ","### ","","",""]
        level_title_list=[""]*7
        # 列表索引值，对应哪一级标题，比如索引值1 对应一级标题
        current_level = 0 # 0 没有标题

        # 根据正则表达式，一行一行匹配
        for line in lines:
            # 判断如果代码块，有#不作为标题处理
            if line.strip() in ("```", "~~~"):
                # True
                in_code = not in_code
            # 如果不是代码块，并且是标题
            # if True => 执行  if False =》 不执行
            if not in_code and title_rule.match(line):
                # 遇到标题，把标题前面临时存储内容合并
                content = "\n".join(temp)
                if new_title or content:
                    parent_title = ""
                    # 找到当前标题 在列表里面上一个索引值的标题
                    for lv in range(current_level-1,0,-1):
                        if level_title_list[lv]:
                            parent_title = level_title_list[lv]
                            break
                    # 如果父标题，添加默认值
                    if not parent_title:
                        parent_title = new_title if new_title else file_title

                    res.append({
                        "title": new_title,
                        "body": content,
                        "file_title": file_title,
                        "parent_title": parent_title,
                    })

                # 把标题放到列表里面
                # 获取标题数据
                match_obj = title_rule.match(line)
                if match_obj:
                    # 获取标题内容 ，获取标题#数量，根据数量放到列表不同位置
                    # #  ["","# 内容","## 内容","### ","","",""]
                    level = len(match_obj.group(1))
                    current_level = level
                    level_title_list[level] = line

                    # 当前列表存储标题索引 后面索引值清空，避免层次混乱
                    for lv in range(current_level+1,7):
                        level_title_list[lv] = ""
                # 记录当前标题
                new_title = line
                # 临时存储列表清空
                temp = []
            else:
                temp.append(line)
        # 因为最后操作之后，没有标题了，把最后一段内容单独处理
        last_content = "\n".join(temp)
        if new_title or last_content:
            parent_title = ""
            for lv in range(current_level-1,0,-1):
                if level_title_list[lv]:
                    parent_title = level_title_list[lv]
                    break
            if not parent_title:
                parent_title = new_title if new_title else file_title

            res.append({
                "title": new_title,
                "body": last_content,
                "file_title": file_title,
                "parent_title": parent_title,
            })
        return res

    # md格式标准化
    def get_input_param(self,
              state: ImportGraphState)->Tuple[str,str]:
        # 获取md内容
        md_content = state.get('md_content')
        # \r \r\n => \n
        if md_content:
            md_content = (md_content.replace("\r\n", "\n")
                          .replace("\r", "\n"))
        # 获取file_title
        file_title = state.get('file_title')
        return md_content,file_title

if __name__ == '__main__':
    setup_logging()
    doc_split_node = DocumentSplitNode()
    # 构建md_content
    file_path = r"D:\dev\6W100-整本手册\auto\6W100-整本手册.md"
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    # 构建数据
    state = {
        "file_title": "6W100-整本手册",
        "md_content": file_content,
        "file_dir": r"D:\dev\6W100-整本手册\auto"
    }

    doc_split_node.process(state)
