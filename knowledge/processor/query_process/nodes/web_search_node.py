import asyncio
import json
from typing import Tuple, List, Dict, Any

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

# 网络搜索节点 mcp方式
class WebSearchNode(BaseNode):
    # 外层同步process
    def process(self, state: QueryGraphState) -> QueryGraphState:
        return asyncio.run(self._async_process(state))

    # 内部真实异步逻辑
    async def _async_process(self,
                             state: QueryGraphState) -> QueryGraphState:
        # 1 参数校验
        rewritten_query = self.validate_param(state)

        # 2 调用mcp工具得到工具返回结果
        try:
            mcp_result = await self.execute_web_search_mcp(rewritten_query)
        except Exception as exc:
            # Web 搜索是增强分支，不应因外部 MCP 故障中断本地知识库问答。
            self.logger.warning("Web/MCP 搜索不可用，降级为本地检索: %s", exc)
            return {}
        if not mcp_result:
            return {}

        return {"web_search_docs":mcp_result}

    # 调用mcp工具得到工具返回结果
    async def execute_web_search_mcp(self,
                 rewritten_query:str)->List[Dict[str,Any]]:
        # 建立和mcp服务器连接
        mcp_client = None

        # 建立连接需要header信息
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json"
        }

        try:
            # 创建连接对象
            mcp_client = MCPServerStreamableHttp(
                name="通用搜索",
                params={"url": self.config.mcp_dashscope_base_url,
                        "headers": headers,
                        },
                cache_tools_list=True,
            )

            # 建立连接
            await mcp_client.connect()

            # 调用mcp服务器工具
            execute_tool_result = await mcp_client.call_tool(
                # mcp服务器工具名称
                tool_name="bailian_web_search",
                arguments={
                    "query": rewritten_query,
                    "count": 3
                }
            )
            print("=="*50)
            print(execute_tool_result)
            print("==" * 50)
            if not execute_tool_result:
                return []

            # 反序列化
            content_text:str = execute_tool_result.content[0].text

            data:Dict[str,Any] = json.loads(content_text)

            # 列表
            pages = data.get('pages')
            # 从pages列表获取具体数据，snippet答案 title标题问题  url网页地址
            # 列表 封装最终数据
            search_result = []
            for page in pages:
                snippet = page.get('snippet',"")
                title = page.get('title',"")
                url = page.get('url',"")
                search_result.append({
                    "snippet": snippet,
                    "title": title,
                    "url": url
                })
            return search_result
        except Exception as e:
            raise e
        finally:
            if mcp_client:
                await mcp_client.cleanup()

    # 参数校验
    def validate_param(self,
                       state: QueryGraphState):
        rewritten_query = state.get('rewritten_query')
        if not rewritten_query:
            raise ValueError("rewritten_query is empty")
        return rewritten_query

if __name__ == "__main__":
    state = {
        "rewritten_query": "关于H3C LA2608，如何使用？",
        "item_names": ["H3C LA2608 室内无线网关"],
    }

    web_search = WebSearchNode()
    result = web_search.process(state)
    print(result)
