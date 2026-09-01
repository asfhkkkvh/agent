"""
OmniRAG — 多 Agent LangGraph 工作流
─────────────────────────────────────────
监督者 Agent 路由 → RAG / Web / 并行双路 → 综合 → 评审（反馈闭环）

核心设计：
- 双路检索（RAG + Web）通过 asyncio.gather 真正并行执行
- 评审 Agent 的 REVISE 反馈会传回综合 Agent，形成真正的评估-修订闭环
- 对话历史进入监督者与综合提示词，checkpoint 记忆真正生效
- 支持流式事件输出（SSE），前端可实时展示 Agent 执行过程
"""

import asyncio
import logging
import os
import uuid
from functools import partial
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.critique_agent import create_critique_agent
from app.agents.rag_agent import create_rag_agent
from app.agents.synthesis_agent import create_synthesis_agent
from app.agents.web_agent import create_web_agent
from app.config import settings

logger = logging.getLogger(__name__)


# ── 状态模式 ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    rag_context: str
    web_context: str
    draft_answer: str
    final_answer: str
    critique: str
    iterations: int
    route: str


# ── LLM 工厂 ──────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0):
    return ChatZhipuAI(
        model=settings.zhipu_model,
        zhipuai_api_key=settings.zhipuai_api_key,
        temperature=temperature,
    )


# ── 对话历史工具 ──────────────────────────────────────────────────────────────

def format_history(messages: List[BaseMessage], window: int | None = None) -> str:
    """取当前问题之前的最近若干轮对话，格式化为文本。

    当前问题本身是 messages 的最后一条，因此被排除在历史之外。
    """
    window = window or settings.history_window
    prior = messages[:-1] if messages else []
    prior = prior[-window:]
    lines = []
    for m in prior:
        role = "用户" if isinstance(m, HumanMessage) else "助手"
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


# ── 监督者节点 ────────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是监督者，负责将研究查询路由到专业智能体。

可用的智能体：
- rag_agent：用于查询已导入的文档/知识库
- web_agent：用于当前事件、最新数据或实时信息
- both：当问题需要同时利用两个信息源时
- synthesis：仅当 RAG 上下文和 Web 上下文都已收集且足够丰富时，才直接综合

重要：如果 RAG 上下文或 Web 上下文为空，绝不能选择 synthesis，必须先去检索（rag_agent / web_agent / both）。
可以结合对话历史判断意图（例如追问"那对比一下呢"应沿用上一轮的信息源）。

分析查询后，仅回复以下之一：rag_agent, web_agent, both, synthesis
"""),
    ("human", """对话历史:
{history}

查询: {query}
已有的 RAG 上下文: {rag_context}
已有的 Web 上下文: {web_context}
"""),
])


async def supervisor_node(state: AgentState) -> AgentState:
    """路由到相应的 Agent。"""
    llm = get_llm()
    chain = SUPERVISOR_PROMPT | llm
    response = await asyncio.to_thread(
        chain.invoke,
        {
            "query": state["query"],
            "history": format_history(state.get("messages", [])),
            "rag_context": state.get("rag_context", ""),
            "web_context": state.get("web_context", ""),
        },
    )
    route = response.content.strip().lower()
    if route not in ["rag_agent", "web_agent", "both", "synthesis"]:
        route = "both"
    logger.info("监督者路由至: %s", route)
    return {**state, "route": route}


# ── Agent 节点（全部异步化，同步阻塞调用放入线程池）──────────────────────────

async def rag_node(state: AgentState) -> AgentState:
    agent = create_rag_agent()
    context = await asyncio.to_thread(agent.run, state["query"])
    return {**state, "rag_context": context}


async def web_node(state: AgentState) -> AgentState:
    agent = create_web_agent()
    context = await asyncio.to_thread(agent.run, state["query"])
    return {**state, "web_context": context}


async def both_node(state: AgentState) -> AgentState:
    """RAG 与 Web 检索真正并行执行（asyncio.gather）。"""
    rag_fut, web_fut = await asyncio.gather(
        rag_node(state),
        web_node(state),
    )
    return {
        **state,
        "rag_context": rag_fut["rag_context"],
        "web_context": web_fut["web_context"],
    }


async def synthesis_node(state: AgentState) -> AgentState:
    """综合生成；把上一版答案与评审反馈一起传入，实现真正的修订闭环。"""
    agent = create_synthesis_agent()
    answer = await asyncio.to_thread(
        agent.run,
        state["query"],
        state.get("rag_context", ""),
        state.get("web_context", ""),
        state.get("critique", ""),
        state.get("draft_answer", ""),
        format_history(state.get("messages", [])),
    )
    return {**state, "draft_answer": answer}


async def critique_node(
    state: AgentState,
    max_iterations: Optional[int] = None,
) -> AgentState:
    """评审答案；通过则结束，否则带反馈回到综合节点。

    max_iterations 通过 build_graph 注入（见 build_graph），
    不修改全局 settings，避免并发请求互相影响。
    """
    agent = create_critique_agent()
    critique = await asyncio.to_thread(
        agent.evaluate,
        state["query"],
        state["draft_answer"],
        state.get("rag_context", "") + "\n" + state.get("web_context", ""),
    )
    iterations = state.get("iterations", 0) + 1
    limit = max_iterations if max_iterations is not None else settings.max_iterations
    passed = _critique_passed(critique)
    if passed or iterations >= limit:
        final = state["draft_answer"]
    else:
        final = ""
    logger.info(
        "评审结果: %s（第 %d/%d 轮）",
        "PASS" if passed else "REVISE",
        iterations,
        limit,
    )
    return {
        **state,
        "critique": critique,
        "final_answer": final,
        "iterations": iterations,
    }


def _critique_passed(critique: str) -> bool:
    """判断评审结果是否通过。

    评审 Agent 可能返回两种格式：
    - "GOOD"（直接通过）
    - "评估：… 评分：GOOD"（结论在末尾）

    同时避免 "NOT GOOD" 被误判为通过。
    """
    text = critique.upper().strip()
    if text.startswith("GOOD"):
        return True
    if "NOT GOOD" in text or "不通过" in text:
        return False
    last_line = text.splitlines()[-1] if text.splitlines() else ""
    return "GOOD" in last_line


# ── 路由函数 ───────────────────────────────────────────────────────────────────

def route_supervisor(state: AgentState) -> Literal["rag_node", "web_node", "both_node", "synthesis_node"]:
    route = state.get("route", "both")
    # 代码层保险：如果上下文都为空，绝不允许直接 synthesis，强制走检索
    has_rag = bool(state.get("rag_context", "").strip())
    has_web = bool(state.get("web_context", "").strip())
    if route == "synthesis" and not has_rag and not has_web:
        logger.warning("监督者路由至 synthesis 但上下文为空，强制改为 both")
        route = "both"
    if route == "rag_agent":
        return "rag_node"
    elif route == "web_agent":
        return "web_node"
    elif route == "both":
        return "both_node"
    else:
        return "synthesis_node"


def route_critique(state: AgentState) -> Literal["synthesis_node", "end"]:
    """如果评审标记了问题，则携带反馈循环回综合，否则结束。"""
    if state.get("final_answer"):
        return "end"
    return "synthesis_node"


# ── 图组装 ─────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None, max_iterations: Optional[int] = None):
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_node", web_node)
    graph.add_node("both_node", both_node)
    graph.add_node("synthesis_node", synthesis_node)
    graph.add_node("critique_node", partial(critique_node, max_iterations=max_iterations))

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "rag_node": "rag_node",
            "web_node": "web_node",
            "both_node": "both_node",
            "synthesis_node": "synthesis_node",
        },
    )

    graph.add_edge("rag_node", "synthesis_node")
    graph.add_edge("web_node", "synthesis_node")
    graph.add_edge("both_node", "synthesis_node")

    graph.add_edge("synthesis_node", "critique_node")

    graph.add_conditional_edges(
        "critique_node",
        route_critique,
        {"synthesis_node": "synthesis_node", "end": END},
    )

    return graph.compile(checkpointer=checkpointer)


# ── 公共接口 ───────────────────────────────────────────────────────────────────

def _initial_state(query: str) -> AgentState:
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "rag_context": "",
        "web_context": "",
        "draft_answer": "",
        "final_answer": "",
        "critique": "",
        "iterations": 0,
        "route": "",
    }



def _extract_result(result) -> dict:
    return {
        "final_answer": result.get("final_answer") or result.get("draft_answer"),
        "rag_context": result.get("rag_context", ""),
        "web_context": result.get("web_context", ""),
        "critique": result.get("critique", ""),
        "iterations": result.get("iterations", 0),
        "route": result.get("route", ""),
    }


async def arun_query(
    query: str,
    thread_id: str = "default",
    max_iterations: Optional[int] = None,
) -> dict:
    """异步执行完整的多 Agent 管道。

    max_iterations 用于覆盖评审闭环轮数（评估时传 1 可关闭循环），
    不修改全局 settings。async 调用方请使用本函数而非 run_query。
    """
    db_path = os.path.join(settings.data_dir, "checkpoints.db")
    os.makedirs(settings.data_dir, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer, max_iterations=max_iterations)
        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        result = await graph.ainvoke(_initial_state(query), config=config)
        return _extract_result(result)


def run_query(
    query: str,
    thread_id: str = "default",
    max_iterations: Optional[int] = None,
) -> dict:
    """同步包装：通过 asyncio.run 执行完整管道。

    注意：只能在没有运行中事件循环的上下文调用（如 FastAPI 同步端点、
    脚本）。若在 async 环境中调用会抛 RuntimeError，请改用 arun_query。
    """
    return asyncio.run(arun_query(query, thread_id, max_iterations=max_iterations))


def _node_event(node: str, frag: dict) -> dict | None:
    """把每个节点输出转换为 SSE 友好的轻量事件。"""
    if node == "supervisor":
        return {"type": "status", "step": "route", "detail": f"路由至 {frag.get('route', 'both')}"}
    if node == "rag_node":
        return {
            "type": "status",
            "step": "rag",
            "detail": f"知识库检索完成（{len(frag.get('rag_context', ''))} 字符）",
        }
    if node == "web_node":
        return {
            "type": "status",
            "step": "web",
            "detail": f"网络搜索完成（{len(frag.get('web_context', ''))} 字符）",
        }
    if node == "both_node":
        return {
            "type": "status",
            "step": "both",
            "detail": "双路并行检索完成",
        }
    if node == "synthesis_node":
        return {
            "type": "status",
            "step": "synthesis",
            "detail": f"综合生成完成（第 {frag.get('iterations', 0) + 1} 轮）",
        }
    if node == "critique_node":
        passed = bool(frag.get("final_answer"))
        return {
            "type": "critique",
            "passed": passed,
            "detail": (frag.get("critique") or "")[:300],
        }
    return None


async def stream_query(
    query: str,
    thread_id: str = "default",
    max_iterations: Optional[int] = None,
):
    """流式执行管道，逐步产出事件字典（供 SSE 使用）。

    事件类型：
    - start:    会话开始，携带 thread_id
    - status:   Agent 节点进度（route / rag / web / both / synthesis）
    - critique: 评审结果（passed + detail）
    - final:    最终结果（完整 answer + 上下文）
    """
    thread_id = thread_id or str(uuid.uuid4())
    db_path = os.path.join(settings.data_dir, "checkpoints.db")
    os.makedirs(settings.data_dir, exist_ok=True)

    yield {"type": "start", "thread_id": thread_id}

    last_state = None
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer, max_iterations=max_iterations)
        config = {"configurable": {"thread_id": thread_id}}

        async for update in graph.astream(
            _initial_state(query), config=config, stream_mode="updates"
        ):
            for node, frag in update.items():
                event = _node_event(node, frag)
                if event:
                    yield event
                if node == "critique_node":
                    last_state = frag

    if last_state:
        yield {"type": "final", "result": _extract_result(last_state)}
    else:
        yield {"type": "error", "detail": "工作流未返回结果"}
