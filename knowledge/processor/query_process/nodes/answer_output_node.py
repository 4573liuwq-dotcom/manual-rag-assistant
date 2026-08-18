"""基于精排证据生成最终答案。"""

import re

from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ANSWER_PROMPT
from knowledge.utils.llm_client_util import get_llm_client


class AnswerOutputNode(BaseNode):
    """将精排文档组织为可追溯的上下文，并调用 LLM 生成答案。"""

    name = "answer_output"
    no_evidence_answer = "抱歉，当前知识库中没有找到足够的信息来回答这个问题。"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        user_query = (
            state.get("rewritten_query", "")
            or state.get("original_query", "")
        ).strip()
        if not user_query:
            raise ValueError("用户问题不能为空")

        reranked_docs = state.get("reranked_docs") or []
        context, sources = self.build_context(reranked_docs)
        state["sources"] = sources

        if not context:
            state["prompt"] = ""
            state["answer"] = self.no_evidence_answer
            return state

        prompt = self.build_prompt(
            user_query=user_query,
            context=context,
            history=state.get("history") or [],
            item_names=state.get("item_names") or [],
        )
        state["prompt"] = prompt
        state["answer"] = self.normalize_citations(
            self.call_llm(prompt), source_count=len(sources)
        )
        return state

    @staticmethod
    def normalize_citations(answer: str, source_count: int) -> str:
        """把“资料1/参考资料1”统一为可校验的 ``[资料1]`` 格式。"""
        def replace(match: re.Match) -> str:
            index = int(match.group(1))
            return f"[资料{index}]" if 1 <= index <= source_count else match.group(0)

        return re.sub(r"(?<!\[)(?:参考)?资料\s*(\d+)(?!\])", replace, answer)

    def build_context(
        self, reranked_docs: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """构建有编号且受长度限制的证据上下文。"""
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        total_chars = 0
        max_chars = max(1, self.config.max_context_chars)

        for doc in reranked_docs:
            content = str(doc.get("content") or "").strip()
            if not content:
                continue

            source_index = len(context_parts) + 1
            title = str(doc.get("title") or "未命名资料").strip()
            source = str(
                doc.get("url") or doc.get("chunk_id") or "未知来源"
            ).strip()
            header = (
                f"[资料{source_index}]\n"
                f"标题：{title}\n"
                f"来源：{source}\n"
                "内容："
            )
            remaining = max_chars - total_chars
            if remaining <= len(header):
                break

            content = content[: remaining - len(header)]
            block = header + content
            context_parts.append(block)
            total_chars += len(block)
            sources.append(
                {
                    "index": source_index,
                    "title": title,
                    "source": source,
                    "score": doc.get("score"),
                }
            )

            if total_chars >= max_chars:
                break

        return "\n\n".join(context_parts), sources

    def build_prompt(
        self,
        user_query: str,
        context: str,
        history: List[Any],
        item_names: List[str],
    ) -> str:
        return ANSWER_PROMPT.format(
            context=context,
            history=self.format_history(history),
            item_names="、".join(item_names) or "未确认",
            question=user_query,
        ).strip()

    @staticmethod
    def format_history(history: List[Any], max_chars: int = 2000) -> str:
        """将最近对话转成简洁文本，避免历史无限增长。"""
        lines: List[str] = []
        for message in history[-6:]:
            if isinstance(message, dict):
                role = str(message.get("role") or "unknown")
                content = str(message.get("content") or "").strip()
                if content:
                    lines.append(f"{role}: {content}")
            elif message is not None:
                text = str(message).strip()
                if text:
                    lines.append(text)
        return "\n".join(lines)[-max_chars:] or "无"

    def call_llm(self, prompt: str) -> str:
        llm_client = get_llm_client(model_name=self.config.default_model)
        response = llm_client.invoke(
            [
                SystemMessage(
                    content=(
                        "你是企业知识库问答助手。回答必须忠于提供的参考资料，"
                        "不得用模型记忆补充资料中不存在的事实。"
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        answer = str(getattr(response, "content", "") or "").strip()
        if not answer:
            raise ValueError("大模型没有返回有效答案")
        return answer
