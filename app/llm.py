"""
OmniRAG — 共享 LLM 工厂
────────────────────────────
所有 Agent / 检索器 / 评估器统一从这里构造 ChatZhipuAI，
避免 6 处各自 new 造成参数散落（model / api_key / temperature 不一致）。

用法：
    from app.llm import get_llm
    llm = get_llm(temperature=0.1, max_tokens=1200)
    # 需要 function calling 时：
    llm = get_llm().bind_tools(TOOLS)
"""

from typing import Optional

from langchain_community.chat_models import ChatZhipuAI

from app.config import settings


def get_llm(
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> ChatZhipuAI:
    """构造智谱 GLM 聊天模型。

    - temperature：0=纯事实（检索/路由/评审），0.1=轻微创造性（综合生成）
    - max_tokens：限制输出长度（生成速度直接决定用户等待时长）
    """
    kwargs: dict = {
        "model": settings.zhipu_model,
        "zhipuai_api_key": settings.zhipuai_api_key,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatZhipuAI(**kwargs)
