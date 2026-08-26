"""
OmniRAG — 综合 Agent
──────────────────────────
将 RAG 上下文 + 网络上下文合并为最终的、有依据的答案。
"""

import logging
from dataclasses import dataclass

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深研究分析师，负责生成最终的、结构良好的答案。

你有两个信息来源：
1. 知识库（RAG）：经过筛选的内部文档
2. 网络搜索：实时互联网结果

指令：
- 将两个来源的信息综合为一个连贯、全面的答案
- 以事实准确性为先；在行内标注来源 [RAG: 来源/页码] 或 [Web: URL]
- 标出来源之间的任何矛盾之处
- 使用清晰的 Markdown 格式，善用标题、列表和表格
- 在结尾附上"来源"章节，列出所有引用
- 如有信息缺失，明确指出哪些是未知的
- 如果提供了"评审反馈"，必须逐条解决反馈中提到的问题；
  如果反馈为 GOOD，说明上一版答案已达标，保持其质量和事实，不要无意义改写。

务必全面、准确、专业。"""),
    ("human", """对话历史（供上下文参考，不要重复用户已提过的内容）:
{history}

问题: {query}

--- RAG 上下文 ---
{rag_context}

--- Web 上下文 ---
{web_context}

--- 上一版答案 ---
{previous_answer}

--- 评审反馈 ---
{critique}

请给出你的综合答案:"""),
])


@dataclass
class SynthesisAgent:
    def run(
        self,
        query: str,
        rag_context: str,
        web_context: str,
        critique: str = "",
        previous_answer: str = "",
        history: str = "",
    ) -> str:
        llm = ChatZhipuAI(
            model=settings.zhipu_model,
            zhipuai_api_key=settings.zhipuai_api_key,
            temperature=0.1,
        )
        chain = SYNTHESIS_PROMPT | llm
        response = chain.invoke({
            "query": query,
            "history": history or "（无）",
            "rag_context": rag_context or "未检索到知识库上下文。",
            "web_context": web_context or "未执行网络搜索。",
            "previous_answer": previous_answer or "（无）",
            "critique": critique or "（无）",
        })
        logger.info("Synthesis agent 完成（含评审反馈: %s）", bool(critique))
        return response.content


def create_synthesis_agent() -> SynthesisAgent:
    return SynthesisAgent()
