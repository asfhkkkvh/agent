"""
OmniRAG — RAGAS 量化评估模块
────────────────────────────────────
使用 RAGAS 框架对 RAG 系统进行端到端评估，输出 4 个核心指标：
    - Faithfulness        答案是否忠于检索到的上下文
    - Answer Relevancy    答案是否切题
    - Context Precision   检索的上下文是否精炼
    - Context Recall      检索是否召回了必要信息

评估流程：
    1. 从已导入文档自动生成 N 条 (query, ground_truth) 评测对
    2. 对每条 query 调用 run_query 获取 (answer, context)
    3. 喂给 RAGAS 跑评估，输出指标 + 单条样本详情
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.graph.workflow import run_query

logger = logging.getLogger(__name__)


# ── 评测样本结构 ─────────────────────────────────────────────────────────────

class EvalSample(BaseModel):
    query: str
    ground_truth: str


class EvalDataset(BaseModel):
    samples: List[EvalSample] = Field(default_factory=list)


# ── 评测集自动生成 ───────────────────────────────────────────────────────────

GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是 RAG 评测集生成助手。基于给定文档片段，
生成 {n} 条 (问题, 标准答案) 对用于评估检索增强系统。

要求：
1. 问题要具体、可从给定文档中找到答案（不要凭空提问）
2. 标准答案要简洁准确，1-2 句话
3. 问题类型多样化：事实型 / 比较型 / 推理型 各占一定比例
4. 返回 JSON 格式"""),
    ("human", "文档片段：\n{context}"),
])


def _sample_context(n: int) -> str:
    """从 Qdrant 随机采样一些 chunk 作为生成评测集的素材。

    用 scroll 直接随机读取，而不是用一个固定查询词去相似度检索——
    后者会把采样结果偏向与"概览/简介"相关的文档。
    """
    try:
        from app.rag.ingestion import get_qdrant_client

        client = get_qdrant_client()
        records, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=max(n * 3, 30),
            with_payload=True,
            with_vectors=False,
        )
        random.shuffle(records)
        texts = [
            r.payload.get("page_content", "")
            for r in records
            if r.payload and r.payload.get("page_content")
        ]
        if not texts:
            return ""
        return "\n\n---\n\n".join(t[:800] for t in texts[: n * 2])
    except Exception as e:
        logger.warning("采样文档失败: %s", e)
        return ""


def _retrieve_raw_contexts(query: str, top_k: int = None) -> List[str]:
    """用检索器取回真实命中的 chunk 原文（评估用的 ground truth 上下文）。

    RAGAS 的 Context Precision / Recall 应该衡量"检索到了什么"，
    因此必须使用原始 chunk，而不是 RAG Agent 二次加工后的摘要。
    """
    from app.rag.retriever import HybridRetriever

    try:
        retriever = HybridRetriever(
            top_k=top_k or settings.retrieval_top_k,
            reranker_top_n=settings.reranker_top_n,
        )
        docs = retriever.invoke(query)
        return [d.page_content for d in docs]
    except Exception as e:
        logger.warning("检索原始上下文失败 (%s): %s", query[:30], e)
        return []


def generate_eval_dataset(n: int = None) -> List[EvalSample]:
    """用 LLM 从已导入文档自动生成 N 条评测样本。"""
    n = n or settings.eval_dataset_size
    context = _sample_context(n)
    if not context:
        logger.error("无法从向量库采样文档，请先导入文档。")
        return []

    llm = ChatZhipuAI(
        model=settings.zhipu_model,
        zhipuai_api_key=settings.zhipuai_api_key,
        temperature=0.4,
    )
    structured_llm = llm.with_structured_output(EvalDataset)
    chain = GEN_PROMPT | structured_llm
    try:
        result: EvalDataset = chain.invoke({"n": n, "context": context})
        samples = result.samples[:n]
        logger.info("生成 %d 条评测样本。", len(samples))
        return samples
    except Exception as e:
        logger.error("评测集生成失败: %s", e)
        return []


# ── RAGAS 端到端评估 ──────────────────────────────────────────────────────────

def _build_ragas_dataset(samples: List[EvalSample]) -> Any:
    """构造 RAGAS 输入数据集：对每条样本跑一次完整管道获取答案，
    同时用检索器取回原始 chunk 作为评估上下文。"""
    from datasets import Dataset

    queries, ground_truths = [], []
    answers, contexts = [], []

    for s in samples:
        queries.append(s.query)
        ground_truths.append([s.ground_truth])  # RAGAS 期望 list
        try:
            result = run_query(s.query, thread_id=f"eval-{hash(s.query) & 0xffff}")
            answers.append(result.get("final_answer") or result.get("draft_answer", ""))
            # 用原始检索 chunk 作为评估上下文（方法论上更准确）
            contexts.append(_retrieve_raw_contexts(s.query) or [""])
        except Exception as e:
            logger.warning("样本评估失败 (%s): %s", s.query[:30], e)
            answers.append("")
            contexts.append([""])

    return Dataset.from_dict({
        "question": queries,
        "ground_truths": ground_truths,
        "answer": answers,
        "contexts": contexts,
    })


def run_ragas_evaluation(samples: List[EvalSample] = None) -> Dict[str, Any]:
    """
    跑一次 RAGAS 评估，返回 4 个核心指标 + 单条样本详情。
    samples 不传时自动生成 settings.eval_dataset_size 条。
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        return {"error": f"RAGAS 未安装: {e}"}

    # 1. 生成评测集
    if samples is None:
        samples = generate_eval_dataset()
        if not samples:
            return {"error": "评测集生成失败，请先导入文档"}

    logger.info("开始 RAGAS 评估，样本数：%d", len(samples))

    # 2. 跑 run_query 获取 (answer, context)
    dataset = _build_ragas_dataset(samples)

    # 3. RAGAS 评估
    try:
        # 用智谱 GLM 作为 RAGAS 的 judge LLM + embeddings
        from langchain_community.chat_models import ChatZhipuAI as ChatLLM
        from langchain_community.embeddings import ZhipuAIEmbeddings

        judge_llm = ChatLLM(
            model=settings.zhipu_model,
            zhipuai_api_key=settings.zhipuai_api_key,
            temperature=0,
        )
        judge_emb = ZhipuAIEmbeddings(
            model=settings.zhipu_embedding_model,
            zhipuai_api_key=settings.zhipuai_api_key,
        )

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=judge_llm,
            embeddings=judge_emb,
        )
    except Exception as e:
        logger.exception("RAGAS 评估异常: %s", e)
        return {"error": f"评估失败: {e}"}

    # 4. 整理输出
    scores = {k: float(v) for k, v in result.items()}
    per_sample = []
    for i, s in enumerate(samples):
        per_sample.append({
            "query": s.query,
            "ground_truth": s.ground_truth,
            "answer": dataset[i]["answer"][:200],
        })

    result = {
        "metrics": scores,
        "sample_count": len(samples),
        "samples": per_sample,
        "summary": (
            f"Faithfulness={scores.get('faithfulness', 0):.3f} | "
            f"AnswerRelevancy={scores.get('answer_relevancy', 0):.3f} | "
            f"ContextPrecision={scores.get('context_precision', 0):.3f} | "
            f"ContextRecall={scores.get('context_recall', 0):.3f}"
        ),
    }

    # 4. 自动存档评估报告（data/eval_reports/eval_<时间戳>.json）
    try:
        report_dir = Path(settings.eval_report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["report_path"] = str(report_path)
        logger.info("评估报告已保存: %s", report_path)
    except Exception as e:
        logger.warning("保存评估报告失败: %s", e)

    return result


if __name__ == "__main__":
    # 命令行直接跑评估
    import json
    result = run_ragas_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2))
