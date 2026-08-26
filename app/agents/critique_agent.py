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
from dataclasses import dataclass

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个严格的事实核查与质量评估智能体。

按照以下标准评估给定答案：

1. 忠实性：每个论断都有提供的上下文支撑（无幻觉）
2. 完整性：原始问题被完整回答
3. 准确性：数字、日期和事实与上下文一致
4. 清晰度：结构良好、可读性强、专业规范

评分规则：
- 如果所有标准都通过 → 精确回复：GOOD
- 如果任何标准未通过 → 回复：REVISE: [具体的改进指令]
  清晰列出每个问题，以便综合智能体进行修正。

务必严格。"GOOD"意味着你愿意为这个答案的准确性担保。"""),
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

        llm = ChatZhipuAI(
            model=settings.zhipu_model,
            zhipuai_api_key=settings.zhipuai_api_key,
            temperature=0,
        )
        chain = CRITIQUE_PROMPT | llm
        response = chain.invoke({
            "query": query,
            "answer": answer,
            "context": context[:4000],  # Token 预算
        })
        critique = response.content.strip()
        logger.info("Critique result: %s", critique[:100])
        return critique


def create_critique_agent() -> CritiqueAgent:
    return CritiqueAgent()
