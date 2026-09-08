#!/usr/bin/env python3
"""
OmniRAG — 黄金集候选生成脚本（方法一：从知识库文档提炼）
────────────────────────────────────────────────────────────
从知识库文档 chunks 提炼候选 (query, ground_truth, source)，
输出 JSON 候选文件，供【人工校验/修改】后导入正式黄金集。

为何是"辅助而非全自动"：
- 问题由 LLM 从 chunk 生成、答案由 LLM 从 chunk 提取，
  直接全量入库会造成"自引用失真"（自己考自己，分数虚高）。
- 因此本脚本只产出候选，人工需要核对答案准确性后再合并进
  data/golden_set.json（人工校验是评估可信度的底线）。

用法：
  python scripts/build_golden_set.py --limit 20
  python scripts/build_golden_set.py --limit 10 --output data/golden_set.candidates.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.llm import get_llm

logger = logging.getLogger(__name__)

# 从 chunk 提炼 问题+参考答案（答案只允许来自片段内容，防编造）
QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一名 RAG 黄金集构建助手。给定一段知识库文档片段，生成一个代表该片段核心信息的问题，并给出基于该片段的参考答案。

要求：
- 问题要像真实用户会问的，具体而非泛泛（禁止"这段讲什么"这类问题）
- 答案必须只来自该片段内容，禁止编造片段外的信息；保留关键数据/技术名/数字
- 答案用一段话（50~150 字），贴近原文措辞
- 只输出一行 JSON 对象，含两个字段：query（问题）、ground_truth（答案）"""),
    ("human", "文档片段:\n{chunk}\n\n来源: {source}"),
])


def load_chunks(limit: int | None) -> list[dict]:
    """从 Qdrant 读取文档 chunks（含内容与 metadata.source）。

    Qdrant 云（南美节点）连接不稳定，scroll 加 3 次退避重试。
    """
    import time

    from app.rag.ingestion import get_qdrant_client

    client = get_qdrant_client()
    last_exc = None
    records = []
    for attempt in range(3):
        try:
            records, _ = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=limit or 1000,
                with_payload=True,
                with_vectors=False,
            )
            break
        except Exception as e:
            last_exc = e
            logger.warning("scroll 第 %d 次失败: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))  # 递增退避

    if not records and last_exc is not None:
        logger.error("scroll 3 次重试均失败: %s", last_exc)
        return []

    chunks = []
    for r in records:
        payload = r.payload or {}
        meta = payload.get("metadata") or {}
        content = payload.get("page_content") or ""
        if not content or len(content) < 80:  # 过短的块不适合生成问题
            continue
        chunks.append({
            "content": content,
            "source": meta.get("source", "unknown"),
        })
    logger.info("读取到 %d 个可用 chunk", len(chunks))
    return chunks


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中容错提取 JSON 对象。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _is_duplicate(query: str, seen: list[str], threshold: float = 0.85) -> bool:
    """用 embedding 余弦相似度判断新问题是否与已有问题重复。"""
    from app.rag.ingestion import get_dense_embeddings

    emb = get_dense_embeddings()
    import numpy as np

    try:
        q_vec = np.asarray(emb.embed_query(query), dtype=float)
        q_norm = np.linalg.norm(q_vec)
        for prev in seen[-20:]:  # 只与最近 20 条比较，控制成本
            p_vec = np.asarray(emb.embed_query(prev), dtype=float)
            p_norm = np.linalg.norm(p_vec)
            if q_norm == 0 or p_norm == 0:
                continue
            sim = float((q_vec @ p_vec) / (q_norm * p_norm))
            if sim >= threshold:
                return True
    except Exception as e:
        logger.warning("去重计算失败（不影响主流程）: %s", e)
    return False


def build_candidates(limit: int | None, output: str) -> int:
    """主流程：读 chunks → LLM 生成候选 → 去重 → 写文件。"""
    chunks = load_chunks(limit)
    if not chunks:
        logger.error("知识库无可用 chunk，请先导入文档")
        return 1

    llm = get_llm(temperature=0.3, max_tokens=300)  # 少量随机性，生成多样问题
    chain = QUESTION_PROMPT | llm

    candidates: list[dict] = []
    seen_queries: list[str] = []
    failed = 0

    for i, c in enumerate(chunks, 1):
        try:
            resp = chain.invoke({
                "chunk": c["content"][:1500],  # 截断过长 chunk，控制 token
                "source": c["source"],
            })
            data = _extract_json(resp.content if hasattr(resp, "content") else str(resp))
            if not data or not data.get("query") or not data.get("ground_truth"):
                failed += 1
                continue
            query = data["query"].strip()
            if _is_duplicate(query, seen_queries):
                continue
            seen_queries.append(query)
            candidates.append({
                "query": query,
                "ground_truth": data["ground_truth"].strip(),
                "source": c["source"],
            })
        except Exception as e:
            logger.warning("chunk %d 生成失败: %s", i, e)
            failed += 1

        if (i % 5 == 0) or (i == len(chunks)):
            logger.info("进度 %d/%d，已生成 %d 条候选", i, len(chunks), len(candidates))

    payload = {
        "_comment": (
            f"黄金集候选（{datetime.now().strftime('%Y-%m-%d %H:%M')} 由 "
            "scripts/build_golden_set.py 从知识库生成，LLM 自动提炼）。"
            "【必须人工校验】答案可能含 LLM 偏差，核对后再合并进 golden_set.json。"
        ),
        "samples": candidates,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        "完成：生成 %d 条候选（失败 %d），已写入 %s",
        len(candidates), failed, output,
    )
    return 0


if __name__ == "__main__":
    from datetime import datetime

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="黄金集候选生成（从知识库提炼）")
    parser.add_argument("--limit", type=int, default=None, help="最多处理的 chunk 数")
    parser.add_argument(
        "--output", type=str, default="data/golden_set.candidates.json",
        help="候选输出路径",
    )
    args = parser.parse_args()
    sys.exit(build_candidates(args.limit, args.output))
