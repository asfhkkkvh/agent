"""
OmniRAG — RAG Agent
────────────────────
专业的 Agent，从混合知识库中检索信息，
并为综合 Agent 格式化上下文。
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.llm import get_llm
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个精准的文档检索研究智能体。

根据下方检索到的上下文片段，提取并整理所有能回答查询的相关信息。
务必全面——包括事实、数据、表格和关联关系。
如果上下文不足以回答，请明确说明。

输出要求（响应速度优先，生成速度直接决定用户等待时长）：
- 输出精炼的关键事实摘要，控制在 250 字以内，用简洁 Markdown 要点呈现
- 每个事实用 [RAG: 来源/页码] 标注出处
- 不要复述上下文原文，不要堆砌细节，只保留与查询直接相关的信息"""),
    ("human", "查询: {query}\n\n检索到的上下文:\n{context}"),
])


@dataclass
class RAGAgent:
    retriever: HybridRetriever

    def run(self, query: str) -> str:
        _t = time.perf_counter()
        # 检索（新版 langchain_core BaseRetriever 用 invoke，旧版 get_relevant_documents 已移除）
        docs = self.retriever.invoke(query)
        logger.info("RAG agent 检索耗时 %.1fs", time.perf_counter() - _t)
        if not docs:
            logger.warning("RAG agent: 未检索到文档: %s", query)
            return "知识库中未找到相关文档。"

        # 格式化上下文
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            score = meta.get("rerank_score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
            context_parts.append(
                f"[Doc {i}] Source: {meta.get('source', 'unknown')} | "
                f"Page: {meta.get('page', '?')} | "
                f"Type: {meta.get('content_type', 'text')} | "
                f"Score: {score_str}\n\n"
                f"{doc.page_content}"
            )
        context = "\n\n" + "─" * 60 + "\n\n".join(context_parts)

        # LLM 综合检索到的文档（统一走 app.llm.get_llm 工厂）
        llm = get_llm(temperature=0, max_tokens=300)  # 限制摘要长度：glm 生成约 35-45 字符/s
        chain = RAG_PROMPT | llm
        _t = time.perf_counter()
        response = chain.invoke({"query": query, "context": context})
        logger.info("RAG agent LLM 综合耗时 %.1fs", time.perf_counter() - _t)
        logger.info("RAG agent 完成，查询: %s", query[:60])
        return response.content


@lru_cache
def create_retriever(
    top_k: int | None = None,
    reranker_top_n: int | None = None,
) -> HybridRetriever:
    """共享混合检索器工厂：rag_agent 与 MCP rag_search 工具复用同一套参数。

    - top_k：候选池大小（重排前的召回数）
    - reranker_top_n：重排后保留的最终条数
    缺省时取 settings 默认值。lru_cache 保证同参数只构造一次。
    """
    return HybridRetriever(
        top_k=top_k or settings.retrieval_top_k,
        reranker_top_n=reranker_top_n or settings.reranker_top_n,
        use_reranking=True,
        use_filter_extraction=settings.use_filter_extraction,
    )


@lru_cache
def create_rag_agent() -> RAGAgent:
    return RAGAgent(retriever=create_retriever())
