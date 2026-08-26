"""
OmniRAG — 知识图谱模块
────────────────────────────────────
使用智谱 GLM 从文档中抽取实体+关系，
存入 Neo4j 图数据库，支持多跳图检索。

与 Qdrant 向量检索并行，构成"向量 + 图谱"三路混合召回：
    Dense(智谱 embedding) + Sparse(BM25) + KG(Neo4j)
"""

import logging
from typing import List, Optional

import jieba
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


# ── LLM 抽取的实体/关系结构 ──────────────────────────────────────────────────

class Entity(BaseModel):
    name: str = Field(..., description="实体名称（规范化形式）")
    type: str = Field("Concept", description="实体类型：Person/Org/Concept/Location/Tech 等")
    description: str = Field("", description="实体简要描述")


class Relation(BaseModel):
    head: str = Field(..., description="头实体名称")
    relation: str = Field(..., description="关系名称（如：开发了/属于/涉及/包含）")
    tail: str = Field(..., description="尾实体名称")


class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)


# ── 抽取 Prompt ──────────────────────────────────────────────────────────────

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是知识图谱构建助手。从给定文本中抽取实体和关系，用于构建 Neo4j 知识图谱。

要求：
1. 实体名称规范化（"GPT-4"、"GPT4" 统一为 "GPT-4"）
2. 只抽取明确提及的事实，不要编造
3. 每段文本最多抽 8 个实体、6 个关系
4. 关系名要简短动词式（"开发了"、"属于"、"使用"）
5. 跨段通用的实体也要抽（人名、组织名、技术名）"""),
    ("human", "文本：\n{text}"),
])


# ── Neo4j 连接（懒加载）──────────────────────────────────────────────────────

_driver = None


def get_driver():
    """获取 Neo4j driver 单例。失败时返回 None（降级为无 KG 检索）。"""
    global _driver
    if _driver is None:
        try:
            from neo4j import GraphDatabase
            _driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
            # 验证连接
            _driver.verify_connectivity()
            logger.info("Neo4j 连接成功: %s", settings.neo4j_uri)
        except Exception as e:
            logger.warning("Neo4j 连接失败（%s），KG 检索将降级跳过。", e)
            _driver = None
    return _driver


# ── 图谱构建 ─────────────────────────────────────────────────────────────────

def _extract_with_llm(text: str) -> ExtractionResult:
    """调用智谱 GLM 抽取实体+关系。"""
    llm = ChatZhipuAI(
        model=settings.zhipu_model,
        zhipuai_api_key=settings.zhipuai_api_key,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ExtractionResult)
    chain = EXTRACT_PROMPT | structured_llm
    try:
        return chain.invoke({"text": text[:3000]})  # 限制长度避免超 token
    except Exception as e:
        logger.warning("实体抽取失败: %s", e)
        return ExtractionResult()


def _cypher_merge(tx, entities: List[Entity], relations: List[Relation], source: str):
    """事务函数：MERGE 实体节点 + 关系边。"""
    for e in entities:
        tx.run(
            "MERGE (n:Entity {name: $name}) "
            "SET n.type = $type, n.description = $description, n.source = $source",
            name=e.name, type=e.type, description=e.description, source=source,
        )
    for r in relations:
        tx.run(
            "MATCH (a:Entity {name: $head}), (b:Entity {name: $tail}) "
            "MERGE (a)-[r:REL {type: $rel, source: $source}]->(b)",
            head=r.head, tail=r.tail, rel=r.relation, source=source,
        )


def build_knowledge_graph(documents: List[Document]) -> int:
    """
    从一批 Document 构建/增量更新 Neo4j 知识图谱。
    每个 chunk 抽取一次实体+关系，合并写入。
    返回写入的关系数（粗略进度指标）。
    """
    driver = get_driver()
    if driver is None:
        logger.warning("Neo4j 不可用，跳过知识图谱构建。")
        return 0

    total_relations = 0
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        result = _extract_with_llm(doc.page_content)
        if not result.entities and not result.relations:
            continue
        try:
            with driver.session() as session:
                session.execute_write(_cypher_merge, result.entities, result.relations, source)
            total_relations += len(result.relations)
        except Exception as e:
            logger.warning("写入 Neo4j 失败 (%s): %s", source, e)

    logger.info("知识图谱构建完成：%d 条关系写入。", total_relations)
    return total_relations


# ── 图谱检索 ─────────────────────────────────────────────────────────────────

def kg_retrieve(query: str, top_k: Optional[int] = None) -> List[Document]:
    """
    从 query 中提取关键词，在 Neo4j 中匹配实体，
    返回 1 跳邻居关系拼成的伪文档（供后续 rerank 使用）。

    如果 Neo4j 不可用或无结果，返回空列表（让 retriever 走向量路）。
    """
    if not settings.use_kg_retrieval:
        return []

    driver = get_driver()
    if driver is None:
        return []

    top_k = top_k or settings.kg_top_k
    keywords = _extract_keywords(query)
    if not keywords:
        return []

    cypher = """
    MATCH (n:Entity)
    WHERE any(k IN $keywords WHERE toLower(n.name) CONTAINS toLower(k))
    MATCH (n)-[r:REL]-(m:Entity)
    RETURN n.name AS head, r.type AS rel, m.name AS tail,
           n.description AS head_desc, m.description AS tail_desc
    LIMIT $limit
    """
    try:
        with driver.session() as session:
            records = session.run(cypher, keywords=keywords, limit=top_k).data()
    except Exception as e:
        logger.warning("KG 检索失败: %s", e)
        return []

    docs: List[Document] = []
    for rec in records:
        content = f"{rec['head']} {rec['rel']} {rec['tail']}"
        if rec.get("head_desc"):
            content += f"\n{rec['head']}: {rec['head_desc']}"
        if rec.get("tail_desc"):
            content += f"\n{rec['tail']}: {rec['tail_desc']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "knowledge_graph",
                "content_type": "kg_triple",
                "head": rec["head"],
                "relation": rec["rel"],
                "tail": rec["tail"],
            },
        ))
    logger.info("KG 检索返回 %d 个三元组（query=%s）", len(docs), query[:30])
    return docs


# 常见中文停用词（仅用于 KG 关键词提取，保持轻量）
_STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "与", "及", "或", "对", "从",
    "到", "等", "一个", "什么", "如何", "怎么", "为什么", "请", "我",
    "你", "它", "这个", "那个", "哪些", "多少", "可以", "知道", "里面",
}


def _extract_keywords(query: str, max_keywords: int = 5) -> List[str]:
    """从查询中提取用于图谱匹配的关键词。

    中文文本没有空格，直接 split 会把整句变成一个"关键词"，
    因此使用 jieba 分词后过滤停用词和单字。
    """
    words = [w.strip() for w in jieba.cut(query) if w.strip()]
    keywords = [w for w in words if len(w) > 1 and w not in _STOPWORDS]
    return keywords[:max_keywords]
