"""
OmniRAG 配置 — 通过 pydantic-settings 集中管理设置。
所有值均来自环境变量（参见 .env.example）。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 大语言模型 (LLM) — 智谱 GLM ────────────────────────────────
    zhipuai_api_key: str = Field(..., validation_alias="ZHIPUAI_API_KEY")
    zhipu_model: str = Field("glm-4-flash", validation_alias="ZHIPU_MODEL")
    zhipu_embedding_model: str = Field(
        "embedding-3", validation_alias="ZHIPU_EMBEDDING_MODEL"
    )

    # ── LangSmith 追踪配置 ────────────────────────────────────────
    langchain_tracing_v2: str = Field("true", validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(..., validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        "omnirag-production", validation_alias="LANGCHAIN_PROJECT"
    )
    langchain_endpoint: str = Field(
        "https://api.smith.langchain.com", validation_alias="LANGCHAIN_ENDPOINT"
    )

    # ── Qdrant 向量数据库（免费云服务）───────────────────────
    qdrant_url: str = Field(..., validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field("omnirag_hybrid", validation_alias="QDRANT_COLLECTION")

    # ── Tavily 网页搜索（免费版）────────────────────────────
    tavily_api_key: str = Field(..., validation_alias="TAVILY_API_KEY")

    # ── Neo4j 知识图谱（Aura 免费云版: https://neo4j.io/aura-free）──
    neo4j_uri: str = Field("bolt://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_username: str = Field("neo4j", validation_alias="NEO4J_USERNAME")
    neo4j_password: str = Field("password", validation_alias="NEO4J_PASSWORD")

    # ── 知识图谱 + RAGAS 评估开关 ────────────────────────────
    use_kg_retrieval: bool = Field(True, validation_alias="USE_KG_RETRIEVAL")
    kg_top_k: int = Field(5, validation_alias="KG_TOP_K")
    eval_dataset_size: int = Field(20, validation_alias="EVAL_DATASET_SIZE")

    # ── RAG 检索调优 ─────────────────────────────────────────────
    retrieval_top_k: int = Field(10, validation_alias="RETRIEVAL_TOP_K")
    reranker_top_n: int = Field(5, validation_alias="RERANKER_TOP_N")
    chunk_size: int = Field(1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, validation_alias="CHUNK_OVERLAP")

    # ── 多查询扩展（Multi-Query Retrieval）────────────────────
    use_multi_query: bool = Field(False, validation_alias="USE_MULTI_QUERY")
    multi_query_count: int = Field(3, validation_alias="MULTI_QUERY_COUNT")

    # ── 对话记忆 ──────────────────────────────────────────────
    history_window: int = Field(6, validation_alias="HISTORY_WINDOW")

    # ── Agent 设置 ───────────────────────────────────────────
    max_iterations: int = Field(5, validation_alias="MAX_ITERATIONS")
    agent_temperature: float = Field(0.0, validation_alias="AGENT_TEMPERATURE")

    # ── 应用配置 ──────────────────────────────────────────────
    app_title: str = "OmniRAG — 多智能体混合研究平台"
    data_dir: str = "data"
    eval_report_dir: str = Field(
        "data/eval_reports", validation_alias="EVAL_REPORT_DIR"
    )
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")


settings = Settings()
