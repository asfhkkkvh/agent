"""
OmniRAG — RAGAS 评估模块（4 维 LLM-judge）
────────────────────────────────────────────
用 RAGAS 框架对知识库跑 4 维评估：
  - faithfulness       忠实度：答案关键论断是否被上下文支撑（无幻觉）
  - answer_relevancy   相关性：答案是否切题、完整覆盖问题
  - context_precision  上下文精确率：检索上下文是否包含回答问题所需的信息
  - context_recall     上下文召回率：标准答案中的信息是否被检索到

评测集：data/golden_set.json（人工校验的 query / ground_truth / source）。
评估流程：每条样本跑完整 Agent 管道拿答案 + 检索原始 chunk 作上下文，
构造成 RAGAS SingleTurnSample，evaluate 输出 4 维指标 + 报告存档。

说明：
- judge LLM 用智谱 GLM（与生成同模型），指标用于相对度量迭代效果，
  而非绝对分数；检索层 recall 可用 context_recall 客观反映召回质量。
- 使用独立 eval_checkpoints.db，避免与用户对话的 SQLite 写锁冲突。
"""

# ── vertexai 兼容 shim ──────────────────────────────────────────────────────
# langchain-community 0.4.x 已移除 ChatVertexAI，ragas 0.4.x 仍在
# ragas/llms/base.py 顶层引用它。本项目评估只用智谱 GLM，不碰 vertexai，
# 因此在 import ragas 之前注入占位模块即可绕过 ImportError。
import sys as _sys
import types as _types

if "langchain_community.chat_models.vertexai" not in _sys.modules:
    _shim = _types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = type("ChatVertexAI", (), {})
    _sys.modules["langchain_community.chat_models.vertexai"] = _shim

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.graph.workflow import run_query
from app.llm import get_llm

logger = logging.getLogger(__name__)

# 评估使用独立的 checkpoint 数据库，避免与用户对话的 SQLite 写锁冲突
_EVAL_DB = os.path.join(settings.data_dir, "eval_checkpoints.db")


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


# ── 检索上下文（RAGAS context_* 指标需要原始 chunk）────────────────────────

def _retrieve_top_k(sample: Dict[str, str], k: int) -> List[str]:
    """用生产检索器取回 top-k 原始 chunk 文本。

    与 run_query 内部的检索是两次独立检索：
    run_query 返回的是 LLM 摘要后的 rag_context（不可用于上下文指标），
    这里直接调 HybridRetriever 拿原始 chunk，供 context_precision/recall 计算。
    """
    from app.rag.retriever import HybridRetriever

    try:
        retriever = HybridRetriever(
            top_k=k,
            reranker_top_n=k,
            use_filter_extraction=False,
            use_multi_query=False,
        )
        docs = retriever.invoke(sample["query"])
        return [d.page_content for d in docs]
    except Exception as e:
        logger.warning("检索失败 (%s): %s", sample["query"][:30], e)
        return []


# ── RAGAS 主流程 ───────────────────────────────────────────────────────────

def run_evaluation(
    samples: Optional[List[Dict[str, str]]] = None,
    k: Optional[int] = None,
    max_iterations: int = 1,
) -> Dict[str, Any]:
    """跑一次 RAGAS 4 维评估，返回指标均值 + 单条样本详情。

    samples 不传时从 data/golden_set.json 加载。
    max_iterations：评审闭环轮数。1=关闭闭环（测"一次性生成"基线），
    2=生产配置（闭环开启，评审可 REVISE 回综合重写，最多 2 轮）。
    两组对比可量化"评审修订闭环"对端到端质量的真实贡献。
    """
    samples = samples or load_golden_set()
    if not samples:
        return {"error": "黄金集为空，请先创建 data/golden_set.json"}

    k = k or settings.retrieval_top_k

    # 延迟导入 ragas（shim 已在模块顶部生效），避免拖慢服务启动
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    eval_llm = LangchainLLMWrapper(get_llm(temperature=0))
    from app.rag.ingestion import get_dense_embeddings

    eval_embeddings = LangchainEmbeddingsWrapper(get_dense_embeddings())
    # 注意：用 ragas.metrics 旧版路径（非 collections）——0.4.x 的 collections
    # metrics 强制要求 OpenAI 风格 InstructorLLM，与智谱 LangchainLLMWrapper
    # 不兼容；旧版路径 llm 参数可选，由 evaluate(llm=...) 全局注入。
    # 仅 DeprecationWarning，不影响功能。
    # ragas.metrics 导出的已是实例（Faithfulness 等），直接放入 metrics 列表
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    ragas_samples: list = []
    per_sample_meta: List[Dict[str, Any]] = []

    for s in samples:
        query = s["query"]
        try:
            # max_iterations：评审闭环轮数（1=基线关闭闭环，2=生产开启闭环）
            # db_path：独立 eval 数据库，避免与用户对话写锁冲突
            result = run_query(
                query,
                thread_id=f"eval-{uuid.uuid4().hex[:8]}",
                max_iterations=max_iterations,
                db_path=_EVAL_DB,
            )
            answer = result.get("final_answer") or result.get("draft_answer", "")
            ctxs = _retrieve_top_k(s, k)

            ragas_samples.append(SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=ctxs,
                reference=s.get("ground_truth", ""),
            ))
            per_sample_meta.append({
                "query": query,
                "ground_truth": s.get("ground_truth", ""),
                "source": s.get("source", ""),
                "answer": (answer or "")[:200],
                "context_count": len(ctxs),
            })
        except Exception as e:
            logger.warning("样本评估失败 (%s): %s", query[:30], e)

    if not ragas_samples:
        return {"error": "所有样本评估失败，请查看日志"}

    dataset = EvaluationDataset(samples=ragas_samples)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=1, max_retries=2),
        show_progress=False,
    )

    # 汇总：RAGAS result 的 to_pandas() 每行一个样本，列即各指标名
    df = result.to_pandas()
    metric_names = [m.name for m in metrics]
    means = {
        name: (float(df[name].mean()) if name in df.columns else None)
        for name in metric_names
    }

    # 单条样本指标并入元数据
    for i, meta in enumerate(per_sample_meta):
        row = df.iloc[i] if i < len(df) else None
        if row is not None:
            for name in metric_names:
                meta[name] = float(row[name]) if name in df.columns else None

    metrics_out = {
        "faithfulness": means.get("faithfulness"),
        "answer_relevancy": means.get("answer_relevancy"),
        "context_precision": means.get("context_precision"),
        "context_recall": means.get("context_recall"),
    }
    summary = " | ".join(
        f"{name}={value:.3f}" if value is not None else f"{name}=N/A"
        for name, value in metrics_out.items()
    )

    result_payload: Dict[str, Any] = {
        "metrics": metrics_out,
        "sample_count": len(ragas_samples),
        "samples": per_sample_meta,
        "summary": summary,
    }

    # 存档报告（指标为 None 而非 NaN，JSON 合法，前端可 JSON.parse）
    try:
        report_dir = Path(settings.eval_report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result_payload["report_path"] = str(report_path)
        logger.info("评估报告已保存: %s", report_path)
    except Exception as e:
        logger.warning("保存评估报告失败: %s", e)

    return result_payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_evaluation(), ensure_ascii=False, indent=2))
