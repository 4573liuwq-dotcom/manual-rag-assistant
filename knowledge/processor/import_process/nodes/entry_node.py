import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState

# 定义节点操作类
class EntryNode(BaseNode):
    # 实现BaseNode里面process方法
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 使用日志输出信息
        self.log_step("步骤1","[开始检查文件类型]")
        # 导入文件目录
        file_dir = state["file_dir"]
        # 导入文件路径
        import_file_path = state["import_file_path"]
        # 文件路径获取文件后缀名，是否.pdf  .md
        path = Path(import_file_path)
        # .pdf  .md
        suffix = path.suffix.lower()
        # 判断
        if suffix == ".pdf":
            self.log_step("pdf","[pdf检查通过]")
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif suffix == ".md":
            self.log_step("md","[md检查通过]")
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            self.log_step("other","[检查不通过]")
            raise ValidationError(f"类型{suffix}检查不通过")
        # 获取文件标题
        file_title = path.stem
        state["file_title"] = file_title
        return state

if __name__ == '__main__':
    # 日志初始化
    setup_logging()

    # 构建字典数据
    enty_state = {
        "file_dir":r"D:\dev\test",
        "import_file_path":r"D:\6W100-整本手册.pdf",
    }

    # EntryNode实例化,执行父类里面 __init__方法
    entry_node = EntryNode()
    # 调用实例，自动执行父类里面 __call__，__call__方法帮调用process方法
    res = entry_node(enty_state)
    print(json.dumps(res, ensure_ascii=False, indent=4))