import base64
import json
import os
import re
from pathlib import Path
from typing import Tuple, List

from openai import OpenAI
from PIL import Image

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.minio_util import get_minio_client

# md里面图片处理类
class MdImageNode(BaseNode):
    def process(self, state:ImportGraphState) -> ImportGraphState:
        # 1 根据上一个节点返回state里面md，获取到md
        # 获取md文档内容
        # 获取图片文件
        # md_content：md内容
        # md_path: md文件路径
        # image_path: 图片路径
        md_content,md_path,image_path = self.get_md_content(state)

        cache_path = md_path.with_suffix(md_path.suffix + ".image-cache.json")
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            state["md_content"] = cached.get("md_content", md_content)
            state["image_count"] = int(cached.get("image_count", 0))
            return state

        # 1.1 判断md是否存在图片
        if not image_path.exists():
            state['md_content'] = md_content
            state['image_count'] = 0
            return state

        # 2 根据图片 + 图片前后内容，获取上下文内容(复杂)
        target_images_context = self.get_images_context(image_path,md_content)

        # 3 根据图片本身 + 上下文内容 调用视觉模型 VLM 生成图片摘要
        # List[(图片1上下文内容),(图片2上下文内容)]
        images_summaries = self.create_image_summary(md_path,
                              target_images_context)

        # 4 获取图片，把图片上传到文件存储服务器 minio，返回文件访问地址
        # 5 更新md里面图片内容数据，包含摘要+文件路径
        # ![这是一个凶猛的老虎](http://192.168.1.1/a/b/laohu.jpg)
        new_md_content = self.upload_img_update_md(md_path,md_content,images_summaries,target_images_context)
        cache_path.write_text(
            json.dumps(
                {"md_content": new_md_content, "image_count": len(target_images_context)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # 6 更新state数据返回
        state['md_content'] = new_md_content
        state['image_count'] = len(target_images_context)
        return state

    # 1 获取md内容，图片路径
    # 根据上一个节点返回state里面md，获取到md
    # 获取md文档内容
    # 获取图片文件
    # md_content：md内容
    # md_path: md文件路径
    # image_path: 图片路径
    def get_md_content(self,state:ImportGraphState) -> Tuple[str, Path, Path]:
        # 获取md_path
        md_path = state['md_path']
        md_path_obj = Path(md_path)
        # md是否存在
        if not md_path_obj.exists():
            raise FileNotFoundError("md文件不存在")

        # 获取md文件内容
        with open(md_path_obj,"r",encoding="utf-8") as f:
            md_content = f.read()

        # 获取图片路径
        # md_path_obj  c:\a\b\test.md  => c:\a\b\images
        # md_path_obj.parent => c:\a\b\
        image_path = md_path_obj.parent / "images"
        # 返回
        return md_content,md_path_obj,image_path

    # 2 根据md内容，图片路径获取md中图片上下文内容
    # 因为image_path有很多图片，返回很多图片的上下文内容，使用列表作为返回类型
    # List[(图片1上下文内容),(图片2上下文内容)]
    # List[("图片名称1","图片路径1",("图片前一个标题","图片上文","图片下文")),
    #       ("图片名称2","图片路径2",("图片前一个标题","图片上文","图片下文"))]
    def get_images_context(self,
                           image_path:Path,
        md_content:str) -> List[Tuple[str, str, Tuple[str, str,str]]]:

        # 最终数据封装
        target_images_context = []
        # 1 根据图片路径 image_path 获取路径所有图片
        #  list[str]
        for image_name in os.listdir(image_path):
            # 2 判断图片是否标准格式图片（后缀名）
            # abc.jpg  wwww.png
            # ('abc','.jpg')
            file_ext = os.path.splitext(image_name)[1]
            #
            config = get_config()
            if file_ext not in config.image_extensions:
                # 跳出本次循环
                continue

            image_file = image_path / image_name
            try:
                with Image.open(image_file) as image:
                    width, height = image.size
                if width * height < config.image_min_area:
                    continue
            except OSError:
                continue

            # 3 把复合格式要求图片，生成上下文内容，抽取方法
            # 参数 图片名称  和  md内容
            # 返回：图片对应上下文数据
            # list
            # [("图片前一个标题","图片上文","图片下文")]
            img_context = self.build_image_context(md_content,image_name)

            # 把每个图片生成上下文内容，封装到最终列表里面
            if not img_context:
                continue
            # 获取唯一一个上下文内容
            single_img_context = img_context[0]
            # 放到最终列表
            # ("图片名称1","图片路径1",("图片前一个标题","图片上文","图片下文"))
            target_images_context.append((image_name,
                                          image_path,
                                          single_img_context))
        # 返回
        return target_images_context

    # 2-1 根据图片名称 和 md内容生成上下文内容
    # 参数 md_content：md内容
    #     image_name： 图片文件名称
    # 返回：[("图片前一个标题","图片上文","图片下文")]
    def build_image_context(self,
                            md_content:str,
           image_name:str) -> List[Tuple[str, str, Tuple[str, str,str]]]:
        # 根据图片文件名称 到 md内容找图片位置
        # 正则匹配
        re_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        # 把md内容拆分行，一行一行匹配
        # split 返回列表 ["",""]
        md_lines = md_content.split("\n")
        images_context = []
        # 把md_lines列表遍历，一行一行匹配
        # line_index行号0开始 ，line每行内容
        for line_index,line  in enumerate(md_lines):
            # 匹配图片
            if not re_pattern.search(line):
                # 不能匹配，结束当前这次循环，匹配下一行内容
                continue
            # 标题
            head_title = "" # 封装图片上面的标题
            head_index = -1 # 图片上面标题行号

            #  1 2 3 4 5
            # 获取图片上文
            # 思路：从图片这一行，开始向上找，直到找到标题行
            # range(开始位置，结束位置，步长)
            # rang(1,5,1)  从1开始找，每次+1  1  2  3  4  直到5
            # rang(5,-1,-1)  从5开始 每次-1  5 4 3 2 1
            for i in range(line_index-1,-1,-1):
                # 找到标题  在md格式   1-6个#空格
                if re.match(r"^#{1,6}\s+",md_lines[i]):
                    # 获取标题
                    head_title = md_lines[i]
                    head_index = i
                    break
            # 获取图片位置 和 上面标题位置，之间的内容
            # ["0","1","2","3","4"]
            # head_index:标题位置 0
            # line_index：图片位置 4
            # [1,4)  左闭右开
            # pre_content 是图片上文内容
            pre_content = md_lines[head_index+1:line_index]
            # 根据获取图片上文内容，对截取，获取上面
            # "front" 从图片开始向上截取
            final_img_pre_context = self.image_context_limit(pre_content,"front")

            # 获取图片下文
            end_index = len(md_lines)
            for i in range(line_index+1,end_index):
                if re.match(r"^#{1,6}\s+",md_lines[i]):
                    end_index = i
                    break
            # 获取图片，下面标题之间内容
            post_context = md_lines[line_index+1:end_index]
            # 调研方法截取
            # "back" 向下截取
            final_img_post_context = self.image_context_limit(post_context,"back")

            # [("图片前一个标题","图片上文","图片下文")]
            images_context.append((head_title,
                                   final_img_pre_context,
                                   final_img_post_context))
        return images_context

    # 根据传入内容截取，
    # front  back
    def image_context_limit(self,
                      substr_content:list,
            substr_type:str) -> str:

        current_content = []
        final_content = []

        img_pattern = re.compile(r"^!\[.*?\]\(.*?\)$")

        for line in substr_content:
            clean_line = line.strip()
            # 你的核心判断：非空、非图片 → 收集行
            if clean_line and not img_pattern.match(clean_line):
                current_content.append(line)
            else:
                # 空行 / 图片行：保存当前段落并清空临时列表
                if current_content:
                    final_content.append("\n".join(current_content))
                    current_content = []

        # 处理循环结束后最后一段内容
        if current_content:
            final_content.append("\n".join(current_content))

        # substr_content截取内容list遍历
        # for line in substr_content:
        #     clean_line = line.strip()
        #     if not clean_line and not re.match(r"^!\[.*?\]\(.*?\)$",clean_line):
        #         current_content.append(line)
            # if not clean_line:# 空行
            #     if current_content:
            #         final_content.append("\n".join(current_content))
            #         current_content = []
            # else: # 不是空行
            #     # 判断当前行是否图片
            #     if re.match(r"^!\[.*?\]\(.*?\)$",clean_line):
            #         if current_content:
            #             final_content.append("\n".join(current_content))
            #             current_content = []
            #         continue
            #     current_content.append(line)

        # if current_content:
        #     final_content.append("\n".join(current_content))

        # 向上截取
        if substr_type=="front":
            # 反转  1 2 3  =》 3 2 1
            final_content.reverse()

        max_char = 200

        total = 0
        selected = []
        for para in final_content:
            para_len = len(para)
            if (total + para_len) > max_char and selected:
                break
            selected.append(para)
            total += para_len

        if substr_type == "front":
            selected.reverse()

        # 6. 返回上下文
        return "\n\n".join(selected)

    #3 调用视觉大模型生成图片摘要
    # 根据图片本身  +  图片上下文内容  =》 调用视觉模型 =》 生成图片摘要
    # 参数：md路径 Path类型
    #      上下文数据：类型
    # List[("图片名称1", "图片路径1", ("图片前一个标题", "图片上文", "图片下文"))]
    def create_image_summary(self,md_path:Path,
         target_images_context:List[Tuple[str, str, Tuple[str, str, Tuple[str, str, str]]]]):
        # 遍历所有图片上下文数据列表 target_images_context
        # img_name：图片名称 a.jpg
        # img_path: 图片路径  比如  D:\dev\6W100-整本手册\auto\images
        # img_context：当前图片 (前一个标题，上文内容，下文内容)
        summaries = {}
        for img_name,img_path,img_context in target_images_context:
            # 图片路径 d:\a\b\11.jpg
            img_file_path = str(img_path / img_name)
            # 调用方法生成每个图片摘要
            summary = self.get_summary_singleImg(md_path,img_file_path,img_context)
            # 每个图片摘要放到summaries
            # {"a.jpg":"摘要" , "b.jpg":"摘要"}
            summaries[img_name] = summary
        return summaries

    # 生成每个图片摘要，调用vlm视觉模型生成
    # md_path: md路径 Path类型  c:\a\b\123.md => 123.md
    # img_file_path：图片路径
    # img_context：图片上下文信息： (前一个标题，上文内容，下文内容)
    def get_summary_singleImg(self,md_path:Path,
              img_file_path:str,
              img_context:Tuple[str, str, str]):
        # 原则：把图片 + 上下文信息，提交vlm，生成摘要
        front_title,pre_content,post_content = img_context
        # 前一个标题，上文内容，下文内容 => str  \n
        # 创建列表，把三个值放到列表，  join \n
        context_data = []
        if front_title:
            context_data.append(front_title)
        if pre_content:
            context_data.append(pre_content)
        if post_content:
            context_data.append(post_content)
        # final_context 是图片上下文数据字符串
        final_context = "\n".join(context_data)

        # img_file_path获取图片
        # 图片 =》 str
        # 读取图片内容，二进制内容，使用base64把图片转换字符串
        image_data_str = ""
        with open(img_file_path,"rb") as f:
            image_data_str = base64.b64encode(f.read()).decode("utf-8")

        # 上面把所需的数据都准备好了，开始调用vlm视觉模型
        # 创建client
        config = get_config()
        client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base
        )

        # 构建messages
        document_title = md_path.stem
        messages = [
            {"role":"user",
             "content":[
                    {
                        "type": "text",
                        "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                         背景信息：
                             1. 所属文档标题："{document_title}"
                             2. 图片上下文：{final_context}
                             请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                             生成一个精准的中文标题（不要包含"图片"二字）。""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data_str}"
                        }
                    }
                ]
             }
        ]

        # 调用
        response = client.chat.completions.create(
            model=config.vl_model,
            messages=messages,
        )

        # 获取结果
        summary = response.choices[0].message.content.strip()
        return summary

    #4 上传图片到minio，更新md内容
    # md_path:md路径
    # md_content:md内容
    # images_summaries:图片摘要
    # target_images_context：图片上下文件数据
    # List[("图片名称1", "图片路径1", ("图片前一个标题", "图片上文", "图片下文"))]
    def upload_img_update_md(self,
                             md_path:Path,
                             md_content:str,
                             images_summaries,
                             target_images_context):
        # 1 上传图片到minio服务，生成可以访问地址
        # 封装minio访问图片多个地址
        remote_urls = {}
        # 创建minio客户端对象
        minio_client = get_minio_client()
        config = get_config()
        if not minio_client.bucket_exists(config.minio_bucket):
            minio_client.make_bucket(config.minio_bucket)
        # 上传图片，获取上传每个图片信息（图片路径，图片名称等）
        # List[("图片名称1", "图片路径1", ("图片前一个标题", "图片上文", "图片下文"))]
        for img_name,img_path,_ in target_images_context:
            # 本地图片路径  d:/a/b/124.jpg
            img_file_path = str(img_path / img_name)

            # 构建上传到minio服务器文件路径+名称
            # a.jpg
            object_name = f"{md_path.stem}/{img_name}"

            # 调用minio_client方法实现上传
            #  bucket_name: str,
            #  object_name: str,
            #  file_path: str,
            minio_client.fput_object(
                config.minio_bucket,
                object_name,
                img_file_path,
            )

            # 生成图片在minio可以访问地址
            # http://192.168.200.139:9000/knowledge-base-v2/abc/1.jpg
            remote_url = ("http://"
                          + config.minio_endpoint +
                          "/" + config.minio_bucket +
                          "/" + object_name)
            print(f"访问路径：{remote_url}")
            # 把所有图片访问路径放到字典
            # remote_urls = {}  {“a.jpg”:"http://....","b.jpg":"http...."}
            remote_urls[img_name] = remote_url

        # 2 根据图片地址 + 图片摘要 更新md内容里面
        new_md_content = md_content
        # 遍历摘要，得到每一个摘要，根据每一个摘要找到对应文件路径，更新到md对应为止
        # {"a.jpg":"摘要1"，"b.jpg":"摘要2"}
        # {“a.jpg”:"路径1","b.jpg":"路径2"}
        for img_name,img_summary in images_summaries.items():
            # 根据摘要文件名称 获取对应访问路径
            remote_url = remote_urls.get(img_name)
            if not remote_url:
                continue

            # 把摘要和对应路径更新到md内容里面
            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",
                re.IGNORECASE)
            new_md_content = replace_pattern.sub(f"![{img_summary}]({remote_url})",new_md_content)
        return new_md_content

if __name__ == '__main__':
    md_image_node = MdImageNode()
    state = {
        "md_path": r"D:\dev\6W100-整本手册\auto\6W100-整本手册.md"
    }
    md_image_node.process(state)
