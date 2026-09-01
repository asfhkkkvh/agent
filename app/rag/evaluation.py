"""
OmniRAG — 轻量评估模块（黄金集 + 检索召回 + LLM-as-judge）
────────────────────────────────────────────
替代早期基于 RAGAS 的重型评估（依赖 ragas / datasets / pyarrow，
存在 pyarrow MonthDayNano pickle bug、NaN 结果、judge 与生成同模型
自引用失真等问题）。

指标：
- recall_at_k：检索层指标。对每条黄金样本取回 top-k chunk，用智谱
  embedding 计算 ground_truth 与各 chunk 的最大余弦相似度，超过阈值
  记为命中，命中率即 recall@k。直接反映"检索有没有召回该召回的东西"。
- faithfulness：答案是否忠于检索上下文（LLM 打分 0-1）。
- answer_relevancy：答案是否切题（LLM 打分 0-1）。

评测集：data/golden_set.json，人工校验的 (query, ground_truth, source)。
不再由系统自产自销，避免自引用失真。
"""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.graph.workflow import run_query

logger = logging.getLogger(__name__)


# ── 黄金集加载 ─────────────────────────────────────────────────────────────

def load_golden_set(path: Optional[str] = None) -> List[Dict[str, str]]:
    """加载黄金集 JSON。

    支持两种格式：
    - 顶层数组：[{"query": ..., "ground_truth": ..., "source": ...}]
    - 顶层对象：{"_comment": "...", "samples": [...]}（便于加说明字段）
    """
    path = path or settings.eval_golden_path
    p = Path(path)
    if not p.exists():
        logger.error("黄金集不存在: %s（请先创建或设置 EVAL_GOLDEN_PATH）", path)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("黄金集 JSON 解析失败: %s", e)
        return []

    rows = data.get("samples", data) if isinstance(data, dict) else data
    samples = [
        {"query": str(s["query"]), "ground_truth": str(s["ground_truth"]),
         "source": str(s.get("source", ""))}
        for s in rows
        if isinstance(s, dict) and s.get("query") and s.get("ground_truth")
    ]
    logger.info("加载黄金集 %d 条（%s）", len(samples), path)
    return samples


# ── 检索层：recall@k ───────────────────────────────────────────────────────

def _embed(texts: List[str]):
    """批量计算智谱 dense 向量（兼容 1 条与多条）。"""
    from app.rag.ingestion import get_dense_embeddings

    emb = get_dense_embeddings()
    if len(texts) == 1:
        return [emb.embed_query(texts[0])]
    return emb.embed_documents(texts)


def _max_cosine(gt_text: str, chunk_texts: List[str]) -> float:
    """ground_truth 与各 chunk 的最大余弦相似度（向量内积归一化）。"""
    if not chunk_texts:
        return 0.0
    try:
        import numpy as np

        gt_vec = np.asarray(_embed([gt_text])[0], dtype=float)
        chunk_vecs = np.asarray(_embed(chunk_texts), dtype=float)
        gt_norm = np.linalg.norm(gt_vec)
        if gt_norm == 0:
            return 0.0
        sims = (chunk_vecs @ gt_vec) / (
            np.linalg.norm(chunk_vecs, axis=1) * gt_norm + 1e-9
        )
        return float(sims.max())
    except Exception as e:
        logger.warning("相似度计算失败: %s", e)
        return 0.0


def _retrieve_top_k(sample: Dict[str, str], k: int) -> List[str]:
    """用生产检索器取回 top-k 原始 chunk 文本。

    关闭过滤器提取与多查询，保持评估确定性、避免额外 LLM 调用。
    """
    from app.rag.retriever import HybridRetriever

    try:
        retriever = HybridRetriever(
            top_k=k,
            reranker_top_n=k,  # recall@k 针对重排后的 top-k 计算
            use_filter_extraction=False,
            use_multi_query=False,
        )
        docs = retriever.invoke(sample["query"])
        return [d.page_content for d in docs]
    except Exception as e:
        logger.warning("检索失败 (%s): %s", sample["query"][:30], e)
        return []


# ── LLM-as-judge ────────────────────────────────────────────────────────────

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是严格的 RAG 质量评估员。根据给定上下文，对模型回答打分。

打分标准（0-100 整数）：
- faithfulness：回答中的关键论断是否都有上下文支撑（无幻觉）。
  完全支撑=90-100，部分支撑=50-80，明显编造=0-30。
- answer_relevancy：回答是否切题、完整覆盖了问题。
  完全切题=90-100，部分=50-80，答非所问=0-30。

只输出一行 JSON：{{"faithfulness": <0-100>, "answer_relevancy": <0-100>}}
不要输出任何其他文字或 markdown 代码块。"""),
    ("human", """问题: {query}

检索到的上下文:
{context}

模型回答:
{answer}

打分:"""),
])


def _judge(query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """LLM 打分，返回 0-1 的 (faithfulness, answer_relevancy)；失败返回空 dict。"""
    if not answer or not answer.strip():
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    llm = ChatZhipuAI(
        model=settings.zhipu_model,
        zhipuai_api_key=settings.zhipuai_api_key,
        temperature=0,
    )
    chain = JUDGE_PROMPT | llm
    context = "\n\n---\n\n".join(contexts)[:4000] if contexts else "（无上下文）"
    try:
        resp = chain.invoke({
            "query": query,
            "context": context,
            "answer": (answer or "")[:2000],
        })
        content = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            logger.warning("judge 输出无法解析: %s", content[:100])
            return {}
        data = json.loads(m.group(0))
        return {
            "faithfulness": float(data.get("faithfulness", 0)) / 100.0,
            "answer_relevancy": float(data.get("answer_relevancy", 0)) / 100.0,
        }
    except Exception as e:
        logger.warning("judge 打分失败: %s", e)
        return {}


# ── 主流程 ──────────────────────────────────────────────────────────────────

def _fmt(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:.3f}"


def run_evaluation(
    samples: Optional[List[Dict[str, str]]] = None,
    k: Optional[int] = None,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    跑一次轻量评估，返回 3 个指标 + 单条样本详情。

    samples 不传时从 data/golden_set.json 加载。
    """
    samples = samples or load_golden_set()
    if not samples:
        return {"error": "黄金集为空，请先创建 data/golden_set.json"}

    k = k or settings.retrieval_top_k
    hits: List[int] = []
    faiths: List[float] = []
    rels: List[float] = []
    per_sample: List[Dict[str, Any]] = []

    for s in samples:
        query = s["query"]
        try:
            # max_iterations=1：关闭 critique 循环，评估关心检索+生成质量，
            # 不重复修改全局 settings（参数化传入）。
            result = run_query(query, thread_id=f"eval-{uuid.uuid4().hex[:8]}", max_iterations=1)
            answer = result.get("final_answer") or result.get("draft_answer", "")
            ctxs = _retrieve_top_k(s, k)
            hit = 1.0 if _max_cosine(s["ground_truth"], ctxs) >= threshold else 0.0
            judge = _judge(query, answer, ctxs)
        except Exception as e:
            logger.warning("样本评估失败 (%s): %s", query[:30], e)
            answer, hit, ctxs, judge = "", 0.0, [], {}

        hits.append(hit)
        if judge.get("faithfulness") is not None:
            faiths.append(judge["faithfulness"])
        if judge.get("answer_relevancy") is not None:
            rels.append(judge["answer_relevancy"])
        per_sample.append({
            "query": query,
            "ground_truth": s.get("ground_truth", ""),
            "source": s.get("source", ""),
            "answer": (answer or "")[:200],
            "recall_hit": hit,
            "faithfulness": judge.get("faithfulness"),
            "answer_relevancy": judge.get("answer_relevancy"),
        })

    metrics = {
        "recall_at_k": (sum(hits) / len(hits)) if hits else None,
        "faithfulness": (sum(faiths) / len(faiths)) if faiths else None,
        "answer_relevancy": (sum(rels) / len(rels)) if rels else None,
    }
    result: Dict[str, Any] = {
        "metrics": metrics,
        "sample_count": len(samples),
        "samples": per_sample,
        "summary": (
            f"Recall@{k}={_fmt(metrics['recall_at_k'])} | "
            f"Faithfulness={_fmt(metrics['faithfulness'])} | "
            f"AnswerRelevancy={_fmt(metrics['answer_relevancy'])}"
        ),
    }

    # 存档报告（metrics 全部为 None 而非 NaN，JSON 合法，前端可 JSON.parse）
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
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_evaluation(), ensure_ascii=False, indent=2))
