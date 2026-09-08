"""
OmniRAG — 网络搜索 Agent
───────────────────────────
通过 Tavily 执行实时网络搜索（免费版：每月 1000 次请求），
并为综合 Agent 格式化结果。
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.llm import get_llm

logger = logging.getLogger(__name__)

WEB_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个网络研究智能体。你的唯一信息来源是下方的实时搜索结果。

【绝对禁止】
- 禁止使用你自身的训练知识回答问题——你只能从搜索结果中提取信息
- 禁止编造搜索结果中不存在的内容、URL、日期、数字
- 如果搜索结果中没有覆盖某个方面，直接说"搜索结果中未提及"

【输出格式】
- 用带编号的小节组织，每节有 **加粗标题**，标题下展开具体事实
- 每条信息必须给出具体数据：模型名、数字、日期、机构名
- 在每条事实后标注来源 [Web: 序号]
- 如有矛盾信息，一并标出
- 总量控制在 400~800 字

以结构化 Markdown 格式呈现。"""),
    ("human", "查询: {query}\n\n搜索结果:\n{results}"),
])


def tavily_search_robust(query: str, max_results: int = 8) -> dict | list:
    """Tavily 双重搜索保障：先 REST 直连（无代理），失败则临时清代理用 SDK 重试。

    返回原始搜索结果（dict 含 results 字段，或 list）。
    供 WebAgent.run 与 MCP web_search 工具复用——两处共用一份代理绕过逻辑，
    避免其中一处被代理卡死而另一处正常，导致行为不一致。
    """
    import os
    import time

    proxy_keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
        "ALL_PROXY", "all_proxy",
    )
    _saved_proxy = {k: os.environ.get(k) for k in proxy_keys}

    def _do_search():
        """直接调用 Tavily REST API，完全控制 httpx 客户端。"""
        import httpx

        # 显式创建无代理的 httpx 客户端，绕过所有环境变量
        client = httpx.Client(
            proxy=None,  # None = 不走代理
            timeout=30,
        )
        try:
            resp = client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            client.close()

    def _do_search_via_sdk():
        """第二条路：走系统默认网络栈（可能经代理）调 Tavily REST API。

        与 _do_search 唯一区别是不传 proxy=None，让 httpx 使用环境代理。
        不用 TavilySearch.invoke()——它返回格式化字符串无法解析结果条数；
        也不用 APIWrapper.raw_results——pydantic 包装后参数冗长易错。
        """
        import httpx

        client = httpx.Client(timeout=30)
        try:
            resp = client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            client.close()

    def _has_error(raw):
        """Tavily 可能返回 dict（含 error 字段）而非抛异常。"""
        if isinstance(raw, dict):
            if raw.get("error"):
                return True
            if not raw.get("results"):
                return True
        return False

    raw = None
    err = None

    # 第一次：用直接 REST API 调用（无代理）
    try:
        _t = time.perf_counter()
        raw = _do_search()
        logger.info("Tavily REST 返回 %s, 耗时 %.1fs", type(raw).__name__, time.perf_counter() - _t)
        if _has_error(raw):
            logger.warning("Tavily REST 返回错误: %s", raw.get("error", "无结果"))
            raw = None
    except Exception as e:
        err = e
        logger.warning("Tavily REST 调用失败: %s", e)

    # 第二次：清代理后用 SDK 重试
    if raw is None:
        for k in proxy_keys:
            os.environ.pop(k, None)
        try:
            _t = time.perf_counter()
            raw = _do_search_via_sdk()
            logger.info("Tavily SDK 直连返回 %s, 耗时 %.1fs", type(raw).__name__, time.perf_counter() - _t)
            if _has_error(raw):
                logger.warning("Tavily SDK 也返回错误: %s", raw.get("error", "无结果"))
                raw = None
        except Exception as e2:
            err = e2
            logger.error("Tavily SDK 直连也失败: %s", e2)
        finally:
            for k, v in _saved_proxy.items():
                if v is not None:
                    os.environ[k] = v

    if raw is None:
        raise RuntimeError(f"网络搜索失败: {err}")
    # 统一规范化为 list：dict（含 results）→ 取 results 字段；list → 原样。
    # 调用方（WebAgent.run / MCP _web_search）拿到的恒为结果列表。
    if isinstance(raw, dict):
        return raw.get("results", [])
    return raw


@dataclass
class WebAgent:
    max_results: int = 8

    def run(self, query: str) -> str:
        import time

        # 搜索执行与格式化分离：搜索逻辑在模块级 tavily_search_robust（MCP 复用），
        # 这里只负责格式化结果 + LLM 生成结构化摘要。
        try:
            _t = time.perf_counter()
            raw = tavily_search_robust(query, self.max_results)
            logger.info("tavily_search_robust 返回, 耗时 %.1fs", time.perf_counter() - _t)
        except Exception as e:
            logger.error("tavily_search_robust 失败: %s", e)
            return f"网络搜索失败: {e}"

        try:
            # tavily_search_robust 已规范化为 list（dict→results），无需再分型处理
            raw_results = raw if isinstance(raw, list) else []

            logger.info("Tavily 搜索结果: %d 条", len(raw_results))

            if not raw_results:
                return "未找到网络搜索结果。"

            formatted = []
            for i, r in enumerate(raw_results, 1):
                if isinstance(r, dict):
                    formatted.append(
                        f"[Web {i}] {r.get('title', 'No Title')}\n"
                        f"URL: {r.get('url', '')}\n"
                        f"Content: {r.get('content', '')}"
                    )
                else:
                    formatted.append(f"[Web {i}] {str(r)}")
            results_text = "\n\n---\n\n".join(formatted)

            llm = get_llm(temperature=0)
            chain = WEB_PROMPT | llm
            response = chain.invoke({"query": query, "results": results_text})
            logger.info("Web agent 完成，查询: %s", query[:60])
            return response.content

        except Exception as e:
            logger.error("Web agent 后处理错误: %s", e)
            return f"网络搜索结果处理失败: {e}"


@lru_cache
def create_web_agent() -> WebAgent:
    return WebAgent(max_results=5)
