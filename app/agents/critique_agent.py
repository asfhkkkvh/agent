"""
OmniRAG — 评估 Agent
─────────────────────────
对合成后的答案进行评估：
  • 忠实性（基于上下文）
  • 完整性（回答完整问题）
  • 准确性（无幻觉）
  • 清晰度（结构良好）

如果答案通过则返回 "GOOD"，或返回具体的改进指令。
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个事实核查与质量评估智能体。

按照以下标准评估给定答案：

1. 忠实性：论断是否被提供的上下文支撑（有无幻觉）
2. 完整性：原始问题是否被完整回答
3. 准确性：数字、日期和事实是否与上下文一致
4. 清晰度：结构是否清晰、可读

评分规则（严格二分，从宽判定）：
- 如果答案实质正确、无幻觉、问题基本被回答 → 精确回复：GOOD
- 仅当存在【明确的硬性错误】时才回复：REVISE: [具体的改进指令]
  硬性错误只包括：幻觉（编造了上下文没有的信息）、与上下文矛盾的事实错误、
  关键数字/日期错误、或者完全没回答用户的问题。

以下情形一律判 GOOD（不是硬性错误，不得 REVISE）：
- 结构不够美观、篇幅偏长偏短、风格措辞等主观偏好
- 还想补充更多细节/更多来源的增量建议
- "可以引用更权威的文档"这类优化意见
- 与检索上下文略有出入但整体方向正确

倾向：强烈默认 GOOD。只有当你能明确指出具体的事实性错误或关键遗漏时才 REVISE。
拿不准时请判 GOOD——与其无谓地多一轮修订拖慢响应，不如直接通过。"""),
    ("human", """原始问题: {query}

使用的上下文:
{context}

生成的答案:
{answer}

你的评估:"""),
])


@dataclass
class CritiqueAgent:
    def evaluate(self, query: str, answer: str, context: str) -> str:
        if not answer or not answer.strip():
            return "REVISE: 答案为空，请生成完整的回答。"

        llm = get_llm(temperature=0)
        chain = CRITIQUE_PROMPT | llm
        _t = time.perf_counter()
        response = chain.invoke({
            "query": query,
            "answer": answer,
            "context": context[:4000],  # Token 预算
        })
        logger.info("Critique agent LLM 耗时 %.1fs", time.perf_counter() - _t)
        critique = response.content.strip()
        logger.info("Critique result: %s", critique[:100])
        return critique


@lru_cache
def create_critique_agent() -> CritiqueAgent:
    return CritiqueAgent()
