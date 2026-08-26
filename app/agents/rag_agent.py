"""
OmniRAG — RAG Agent
────────────────────
专业的 Agent，从混合知识库中检索信息，
并为综合 Agent 格式化上下文。
"""

import logging
from dataclasses import dataclass

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个精准的文档检索研究智能体。

根据下方检索到的上下文片段，提取并整理所有能回答查询的相关信息。
务必全面——包括事实、数据、表格和关联关系。
如果上下文不足以回答，请明确说明。

以结构化 Markdown 格式回复，使用清晰的章节标题。
每个事实都必须标注来源和页码。"""),
    ("human", "查询: {query}\n\n检索到的上下文:\n{context}"),
])


@dataclass
class RAGAgent:
    retriever: HybridRetriever

    def run(self, query: str) -> str:
        # 检索（新版 langchain_core BaseRetriever 用 invoke，旧版 get_relevant_documents 已移除）
        docs = self.retriever.invoke(query)
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

        # LLM 综合检索到的文档
        llm = ChatZhipuAI(
            model=settings.zhipu_model,
            zhipuai_api_key=settings.zhipuai_api_key,
            temperature=0,
        )
        chain = RAG_PROMPT | llm
        response = chain.invoke({"query": query, "context": context})
        logger.info("RAG agent 完成，查询: %s", query[:60])
        return response.content


def create_rag_agent() -> RAGAgent:
    retriever = HybridRetriever(
        top_k=settings.retrieval_top_k,
        reranker_top_n=settings.reranker_top_n,
        use_reranking=True,
        use_filter_extraction=True,
    )
    return RAGAgent(retriever=retriever)
