"""
OmniRAG — Cross-Encoder 重排序器
──────────────────────────────────
使用 BAAI/bge-reranker-base 对查询-文档对进行评分，
并返回排名前 N 的最相关结果。
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    使用 sentence-transformers 的轻量级 cross-encoder 重排序器。
    模型：BAAI/bge-reranker-base（免费，本地运行）。
    """

    _model = None  # 懒加载单例

    def __init__(self, top_n: int = 5, model_name: str = "BAAI/bge-reranker-base"):
        self.top_n = top_n
        self.model_name = model_name

    def _get_model(self):
        if CrossEncoderReranker._model is None:
            try:
                from sentence_transformers import CrossEncoder
                CrossEncoderReranker._model = CrossEncoder(self.model_name)
                logger.info("已加载重排序器: %s", self.model_name)
            except Exception as e:
                logger.warning("重排序器加载失败（%s），跳过重排序。", e)
                CrossEncoderReranker._model = None
        return CrossEncoderReranker._model

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """根据与查询的相关性对文档进行评分和排序。"""
        model = self._get_model()
        if model is None or not documents:
            return documents[: self.top_n]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        scored = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )

        reranked = []
        for score, doc in scored[: self.top_n]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)

        return reranked
