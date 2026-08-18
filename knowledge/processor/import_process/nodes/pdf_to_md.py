import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import PdfConversionError, FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState

# 模块：pdf转换md
# 1 对参数校验，判断文件是否存在
# 2 使用MinerU工具把pdf转换md （命令行实现）
# 3 获取转换之后md的路径
# 4 返回需要数据 （包含md路径等信息）
class PdfToMd(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1 对参数校验，判断文件是否存在
        import_file_path,file_dir = self.validate_path(state)

        # 已完成转换时直接复用结果，方便失败任务从后续节点继续执行。
        md_path = self.get_md_path(import_file_path, file_dir)
        if Path(md_path).exists():
            state["md_path"] = md_path
            return state

        # 2 使用MinerU工具把pdf转换md
        # 转换返回0 成功
        process_code = self.execute_mineru(import_file_path,file_dir)
        # 判断是否0
        if process_code != 0:
            raise PdfConversionError("mineru转换失败")

        # 3 获取转换之后md路径
        md_path = self.get_md_path(import_file_path,file_dir)

        # 4 返回数据
        state["md_path"] = md_path
        return state

    # 方法3：获取转换之后md路径
    def get_md_path(self,import_file_path:Path,file_dir:Path):
        # 获取文件名称 ，从import_file_path 文件路径 c:\dev\test.pdf
        file_name = import_file_path.stem
        # auto 固定值，实际需要客户端执行查看
        md_path = file_dir / file_name / "auto" / f"{file_name}.md"
        return str(md_path)

    # 方法2：使用MinerU工具把pdf转换md
    # 命令行方式调用本地转换
    def execute_mineru(self,import_file_path:Path,file_dir:Path):
        self.log_step("执行mineru转换过程.....")
        mineru_command = shutil.which("mineru")
        if not mineru_command:
            scripts_dir = Path(sys.executable).parent / "Scripts"
            candidate = scripts_dir / ("mineru.exe" if sys.platform == "win32" else "mineru")
            if candidate.exists():
                mineru_command = str(candidate)
        if not mineru_command:
            raise PdfConversionError("当前 Python 环境中未找到 MinerU 可执行文件")

        # 命令行方式调用本地转换
        # MinerU 3.x 从用户目录下的 mineru.json 读取本地模型路径。
        # 不在节点内覆盖 HF_HOME 或 MODELSCOPE_CACHE。
        cmd = [
            mineru_command, "-p",
            str(import_file_path),
            "-o", str(file_dir),
            "--backend", "pipeline",
            "--method", "auto",
        ]

        # 执行cmd构建命令，创建子进程执行
        proc = subprocess.Popen(
            args=cmd, # 执行命令构建列表
            stdout=subprocess.PIPE,  # 正常执行过程（日志）
            stderr=subprocess.STDOUT, # 错误信息
            errors="replace",  # ignore   replace
            text=True, # 文本形式
            bufsize=1
        )
        # 输出stdout=subprocess.PIPE,每行日志
        for line in proc.stdout:
            self.log_step(f"执行MinerU日志: {line}")

        # 等待子进程操作完成, 返回0成功
        process_code = proc.wait()
        if process_code == 0:
            self.logger.info("执行mineru成功了!!!")
        else:
            self.logger.error("执行mineru失败了...")
        return process_code

    # 方法1： 判断文件是否存在
    # 从state获取文件路径，封装path对象，调用path方法判断
    def validate_path(self,state: ImportGraphState) -> Tuple[Path, Path]:
        # 从state获取文件路径
        # state.get('pdf_path', '')
        pdf_path = state["import_file_path"]
        # 封装path对象
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"文件不存在{pdf_path_obj}")
        # 从state获取文件目录
        file_dir = state["file_dir"]
        if not file_dir:
            # c:\a\b\test.pdf  => c:\a\b
            file_dir = pdf_path_obj.parent
        # # 封装path对象
        file_dir_path = Path(file_dir)
        # 返回文件路径 和  文件目录
        return pdf_path_obj, file_dir_path

if __name__ == "__main__":
    # 日志初始化
    setup_logging()
    # PdfToMd实例
    pdf_to_md_node = PdfToMd()
    # 构建参数
    pdf_to_md_node_state = {
        "import_file_path":r"D:\6W100-整本手册.pdf",
        "file_dir":r"D:\dev\0604"
    }
    # 调用对象的方法
    process_result = pdf_to_md_node.process(pdf_to_md_node_state)
    # 输出
    print(json.dumps(process_result, indent=4,ensure_ascii=False))
