"""
OmniRAG — v2 对话记忆库（长期记忆 P0）
────────────────────────────────────────
把高质量对话提炼为可跨会话检索的记忆条目，存独立 Qdrant 集合。

与文档知识库物理隔离（不同 collection），查询时与文档 RAG 并行召回，
注入综合 prompt 的"长期记忆"区域（作为提示，不覆盖文档事实）。

设计要点（面试可讲）：
- 提炼：critique PASS 后异步触发，LLM 只提取 fact / preference / event 三类，
  过滤寒暄与一次性问题 —— 这是"提炼"与"存档"的本质区别
- 存储：独立 omnirag_memory 集合（纯 dense，复用 embedding-3，1024 维）
- 去重：新候选与已有记忆余弦相似度 > DEDUP_THRESHOLD 则跳过
- 召回：与文档 RAG 并行，几 ms 级开销，不阻塞在线路径
- 隔离：集合级（文档 omnirag_hybrid / 记忆 omnirag_memory）物理独立
"""

import json
import logging
import time
import uuid
from functools import lru_cache
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm

logger = logging.getLogger(__name__)

MEMORY_COLLECTION = "omnirag_memory"
DEDUP_THRESHOLD = 0.92  # 与已有记忆的余弦相似度超过此值视为重复

MEMORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是对话记忆提炼器。从"用户提问 + 助手回答"中提取值得跨会话记住的信息。

只输出一个 JSON 数组，每项格式 {{"content": "...", "type": "fact|preference|event"}}。

type 说明：
- fact：关于用户或其项目的客观事实（如"用户的项目使用 Qdrant 向量库"）
- preference：用户偏好（如"用户希望回答简洁、用中文"）
- event：发生过的重要事件（如"修复了 Qdrant TLS 连接问题"）

规则：
- 只提炼能复用于未来对话的信息；寒暄、一次性问题（如天气、具体某天的新闻）不提炼
- content 用简洁陈述句，以"用户…"开头描述用户侧信息
- 无法提炼时输出空数组 []"""),
    ("human", "用户提问: {query}\n\n助手回答: {answer}"),
])


# ── 存储层：独立集合 + 向量存储 ───────────────────────────────────────────────

def _ensure_memory_collection():
    """幂等创建记忆集合（与文档集合物理隔离，纯 dense 1024 维）。"""
    from qdrant_client import models

    from app.rag.ingestion import get_dense_embeddings, get_qdrant_client

    client = get_qdrant_client()
    if not client.collection_exists(MEMORY_COLLECTION):
        client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=models.VectorParams(
                size=1024,  # embedding-3 维度，与文档集合一致
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("记忆集合已创建: %s", MEMORY_COLLECTION)
    return get_dense_embeddings()


@lru_cache
def _get_memory_store():
    """记忆向量存储（lru_cache 单例）。

    必须复用 get_qdrant_client()（含代理 patch：TLS 1.2 + 重试 + 连接重建），
    否则 QdrantVectorStore 内部新建的 client 不走代理会 SSL 连接失败。
    """
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    from app.rag.ingestion import get_dense_embeddings, get_qdrant_client

    embeddings = _ensure_memory_collection()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=MEMORY_COLLECTION,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,  # 记忆条目是短文本，纯 dense 足够
    )


# ── 写入管道：提炼 → 去重 → 入库 ──────────────────────────────────────────────

def extract_memory_candidates(query: str, answer: str) -> list[dict[str, str]]:
    """LLM 提炼记忆候选（fact / preference / event 三类）。"""
    llm = get_llm(temperature=0, max_tokens=400)
    response = llm.invoke(MEMORY_PROMPT.format_messages(query=query[:500], answer=answer[:2000]))
    text = (response.content or "").strip()
    try:
        # 容错：LLM 可能带 ```json 围栏或前后缀文字
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict) and item.get("content")]
    except Exception:
        logger.warning("记忆提炼输出解析失败，丢弃: %s", text[:100])
        return []


def save_memories(candidates: list[dict[str, Any]]) -> int:
    """去重后写入记忆集合，返回实际写入条数。"""
    if not candidates:
        return 0
    store = _get_memory_store()
    written = 0
    for cand in candidates:
        content = cand.get("content", "").strip()
        if not content:
            continue
        # 去重：与已有记忆的相似度最高条目比较
        try:
            hits = store.similarity_search_with_score(content, k=1)
            if hits and hits[0][1] > DEDUP_THRESHOLD:
                logger.info("记忆去重跳过: %s", content[:40])
                continue
        except Exception:
            pass
        store.add_texts(
            [content],
            metadatas=[{
                "type": cand.get("type", "fact"),
                "memory_id": uuid.uuid4().hex[:12],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }],
        )
        written += 1
    if written:
        logger.info("记忆写入 %d 条", written)
    return written


def extract_and_save_memories(query: str, answer: str) -> int:
    """完整写入管道：提炼 → 去重 → 入库（供后台任务调用）。"""
    _t = time.perf_counter()
    try:
        candidates = extract_memory_candidates(query, answer)
        written = save_memories(candidates)
        logger.info("记忆提炼耗时 %.1fs（候选 %d / 写入 %d）",
                    time.perf_counter() - _t, len(candidates), written)
        return written
    except Exception as e:
        logger.warning("记忆提炼失败（不影响对话）: %s", e)
        return 0


# ── 召回：与文档 RAG 并行 ─────────────────────────────────────────────────────

def recall_memories(query: str, top_k: int = 3) -> list[str]:
    """检索记忆集合，返回记忆条目文本列表。

    记忆是"提示"而非"事实"：冲突时以文档为准（综合 prompt 已声明）。
    任何失败都静默降级为空列表，绝不阻塞在线查询。
    """
    try:
        store = _get_memory_store()
        hits = store.similarity_search_with_score(query, k=top_k)
        return [h[0].page_content for h in hits]
    except Exception as e:
        logger.warning("记忆召回失败（降级为空）: %s", e)
        return []
