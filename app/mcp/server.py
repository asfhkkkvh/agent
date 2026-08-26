"""
OmniRAG — MCP 服务器
─────────────────────
自定义 Model Context Protocol 服务器，暴露以下工具：
  • hybrid_search      — 搜索知识库
  • web_search         — Tavily 实时网络搜索
  • summarise_docs     — 总结检索到的上下文
  • extract_entities   — 对检索文本进行命名实体识别
  • calculate          — 安全的 Python 表达式求值器

独立运行：python -m app.mcp.server
"""

import ast
import logging
import operator
import os
import sys
from typing import Any


def _ensure_pywin32_dlls():
    """Windows 下 pywin32 的 DLL 目录（pywin32_system32）默认不在 sys.path 中。

    兼容两种安装方式：
    - 常规 pip 安装：pywintypes 已可直接导入，本函数无副作用
    - vendored .libs 场景：把 pywin32_system32 加入 sys.path，避免 ImportError
    """
    if sys.platform != "win32":
        return
    for entry in list(sys.path):
        dll_dir = os.path.join(entry, "pywin32_system32")
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except (OSError, ValueError):
                pass
            sys.path.insert(0, dll_dir)
            return


_ensure_pywin32_dlls()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

app = Server("omnirag-mcp")

# ── 工具定义 ──────────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="hybrid_search",
        description=(
            "使用混合 dense+sparse 检索并通过 cross-encoder 重排序来搜索内部知识库。"
            "用于查询已导入文档的问题。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "top_k": {"type": "integer", "default": 5, "description": "返回的结果数量"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="web_search",
        description=(
            "使用 Tavily 搜索实时网络。用于查询当前事件、最新数据、"
            "或知识库中没有的内容。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "网络搜索查询"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="summarise_docs",
        description="将文档内容列表总结为简洁的段落。",
        inputSchema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "需要总结的文档文本块列表",
                },
                "focus": {"type": "string", "description": "关注的方面"},
            },
            "required": ["documents"],
        },
    ),
    Tool(
        name="extract_entities",
        description="从文本中提取命名实体（人物、组织、日期、数字）。",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "需要提取实体的文本"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="calculate",
        description="安全地求值数学表达式。例如 '2 + 2 * 10 / 5'。",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"},
            },
            "required": ["expression"],
        },
    ),
]


# ── 工具处理器 ─────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "hybrid_search":
        return await _hybrid_search(**arguments)
    elif name == "web_search":
        return await _web_search(**arguments)
    elif name == "summarise_docs":
        return await _summarise_docs(**arguments)
    elif name == "extract_entities":
        return await _extract_entities(**arguments)
    elif name == "calculate":
        return await _calculate(**arguments)
    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


# ── 实现 ───────────────────────────────────────────────────────────────────────

async def _hybrid_search(query: str, top_k: int = 5) -> list[TextContent]:
    try:
        from app.rag.retriever import HybridRetriever
        retriever = HybridRetriever(reranker_top_n=top_k)
        docs = retriever.invoke(query)
        if not docs:
            return [TextContent(type="text", text="未找到相关文档。")]

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            score = meta.get("rerank_score", "N/A")
            results.append(
                f"[结果 {i}] 来源: {meta.get('source','未知')} | "
                f"页码: {meta.get('page','?')} | "
                f"类型: {meta.get('content_type','text')} | "
                f"重排序分数: {score}\n\n{doc.page_content}"
            )
        return [TextContent(type="text", text="\n\n---\n\n".join(results))]
    except Exception as e:
        logger.error("hybrid_search 错误: %s", e)
        return [TextContent(type="text", text=f"搜索错误: {e}")]


async def _web_search(query: str, max_results: int = 5) -> list[TextContent]:
    try:
        from tavily import TavilyClient

        from app.config import settings
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=max_results)
        results = []
        for r in response.get("results", []):
            results.append(
                f"**{r.get('title','无标题')}**\n"
                f"URL: {r.get('url','')}\n"
                f"{r.get('content','')}"
            )
        return [TextContent(type="text", text="\n\n---\n\n".join(results) or "无网络搜索结果。")]
    except Exception as e:
        logger.error("web_search 错误: %s", e)
        return [TextContent(type="text", text=f"网络搜索错误: {e}")]


async def _summarise_docs(documents: list, focus: str = "") -> list[TextContent]:
    try:
        from langchain_community.chat_models import ChatZhipuAI
        from langchain_core.prompts import ChatPromptTemplate

        from app.config import settings

        combined = "\n\n---\n\n".join(documents[:8])
        focus_clause = f"请特别关注: {focus}。" if focus else ""

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"简洁地总结以下文档内容。{focus_clause}"),
            ("human", "{text}"),
        ])
        llm = ChatZhipuAI(
            model=settings.zhipu_model,
            zhipuai_api_key=settings.zhipuai_api_key,
            temperature=0,
        )
        response = (prompt | llm).invoke({"text": combined})
        return [TextContent(type="text", text=response.content)]
    except Exception as e:
        return [TextContent(type="text", text=f"总结错误: {e}")]


async def _extract_entities(text: str) -> list[TextContent]:
    try:
        from langchain_community.chat_models import ChatZhipuAI
        from langchain_core.prompts import ChatPromptTemplate

        from app.config import settings

        prompt = ChatPromptTemplate.from_messages([
            ("system", """从文本中提取命名实体。以结构化列表形式返回：
人物: ...
组织: ...
日期: ...
数字/指标: ...
地点: ..."""),
            ("human", "{text}"),
        ])
        llm = ChatZhipuAI(
            model=settings.zhipu_model,
            zhipuai_api_key=settings.zhipuai_api_key,
            temperature=0,
        )
        response = (prompt | llm).invoke({"text": text[:3000]})
        return [TextContent(type="text", text=response.content)]
    except Exception as e:
        return [TextContent(type="text", text=f"实体提取错误: {e}")]


async def _calculate(expression: str) -> list[TextContent]:
    """安全的数学求值器 — 不执行/求值任意代码。"""
    SAFE_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return SAFE_OPS[type(node.op)](_eval(node.operand))
        else:
            raise ValueError(f"不支持的操作: {type(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return [TextContent(type="text", text=f"{expression} = {result}")]
    except Exception as e:
        return [TextContent(type="text", text=f"计算错误: {e}")]


# ── 入口点 ─────────────────────────────────────────────────────────────────────

async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())
