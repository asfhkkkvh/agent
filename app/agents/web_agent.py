"""
OmniRAG — 网络搜索 Agent
───────────────────────────
通过 Tavily 执行实时网络搜索（免费版：每月 1000 次请求），
并为综合 Agent 格式化结果。
"""

import logging
from dataclasses import dataclass

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper

from app.config import settings

logger = logging.getLogger(__name__)

WEB_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个网络研究智能体。根据下方的实时搜索结果，
提取、整理并呈现所有能回答查询的相关信息。

包含以下内容：
- 关键事实和数据
- 来源 URL
- 可获取的发布日期
- 如有矛盾信息，一并标出

以结构化 Markdown 格式呈现。务必全面但简洁。"""),
    ("human", "查询: {query}\n\n搜索结果:\n{results}"),
])


@dataclass
class WebAgent:
    max_results: int = 5

    def run(self, query: str) -> str:
        try:
            tool = TavilySearch(
                max_results=self.max_results,
                api_wrapper=TavilySearchAPIWrapper(tavily_api_key=settings.tavily_api_key),
            )
            # TavilySearch.invoke 接受字符串，不是 {"query": ...}
            raw_results = tool.invoke(query)

            if not raw_results:
                return "未找到网络搜索结果。"

            formatted = []
            for i, r in enumerate(raw_results, 1):
                # 兼容 Tavily 返回 dict 或 str 的情况
                if isinstance(r, dict):
                    formatted.append(
                        f"[Web {i}] {r.get('title', 'No Title')}\n"
                        f"URL: {r.get('url', '')}\n"
                        f"Content: {r.get('content', '')}"
                    )
                else:
                    formatted.append(f"[Web {i}] {str(r)}")
            results_text = "\n\n---\n\n".join(formatted)

            llm = ChatZhipuAI(
                model=settings.zhipu_model,
                zhipuai_api_key=settings.zhipuai_api_key,
                temperature=0,
            )
            chain = WEB_PROMPT | llm
            response = chain.invoke({"query": query, "results": results_text})
            logger.info("Web agent 完成，查询: %s", query[:60])
            return response.content

        except Exception as e:
            logger.error("Web agent 错误: %s", e)
            return f"网络搜索失败: {e}"


def create_web_agent() -> WebAgent:
    return WebAgent(max_results=5)
