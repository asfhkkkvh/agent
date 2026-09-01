"""
OmniRAG — 导入管道
─────────────────────────────
文档解析 → 分块 → 嵌入（dense + sparse）→ 插入 Qdrant 混合集合。

文档解析优先级：
1. Docling（若已安装）：支持 PDF/DOCX 富结构（表格、版面）
2. PyMuPDF（已内置）：PDF 纯文本+表格兜底
3. zipfile 标准库：DOCX 纯文本兜底（零额外依赖）
TXT/Markdown 直接读取。
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # docling 可选：未安装时走 pymupdf/zipfile 兜底
    DocumentConverter = None
from fastembed import SparseTextEmbedding

# 注：langchain_community.embeddings.ZhipuAIEmbeddings 在新版被标记弃用，
# 但独立包 langchain-zhipuai 在镜像源下载受限（403），暂继续使用 community 版本。
# 待网络允许后，迁移为：from langchain_zhipuai import ZhipuAIEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from app.config import settings

logger = logging.getLogger(__name__)


# ── 嵌入模型 ──────────────────────────────────────────────────────────────────

def get_dense_embeddings():
    """智谱 dense 嵌入（embedding-3, 指定 1024 维以匹配 Qdrant collection）。"""
    return ZhipuAIEmbeddings(
        model=settings.zhipu_embedding_model,
        api_key=settings.zhipuai_api_key,  # langchain-community>=0.4 字段从 zhipuai_api_key 改为 api_key
        dimensions=1024,
    )


def get_sparse_embeddings():
    """通过 FastEmbed 实现的 BM25 稀疏嵌入。

    FastEmbed 的 SparseTextEmbedding 用 embed() 方法，但 langchain_qdrant
    的 QdrantVectorStore 期望 langchain Embeddings 接口（embed_documents /
    embed_query），所以包一层适配器。
    """
    from langchain_core.embeddings import Embeddings

    class FastEmbedSparseAdapter(Embeddings):
        """把 FastEmbed SparseTextEmbedding 适配为 langchain Embeddings。"""

        def __init__(self, model_name: str = "Qdrant/bm25"):
            self._model = SparseTextEmbedding(model_name=model_name)

        def embed_documents(self, texts):
            # FastEmbed 的 embed 返回 SparseEmbedding 对象列表
            return list(self._model.embed(texts))

        def embed_query(self, text):
            return next(iter(self._model.query_embed(text)))

    return FastEmbedSparseAdapter(model_name="Qdrant/bm25")


# ── Qdrant 客户端与集合 ───────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """创建 Qdrant 客户端。

    国内环境直连 Qdrant Cloud（南美 AWS）会被 GFW 干扰导致 SSL EOF。
    如果检测到系统代理（HTTPS_PROXY / HTTP_PROXY），则 patch qdrant_client
    内部的 ApiClient.send_inner，强制用配了代理的 httpx.Client 发请求。
    """
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        _patch_qdrant_proxy(proxy)

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
        port=443,
        check_compatibility=False,
    )


# 代理 patch 只执行一次
_proxy_patched = False


def _patch_qdrant_proxy(proxy_url: str):
    """Patch qdrant_client.http.api_client.ApiClient.send_inner 使用代理。"""
    global _proxy_patched
    if _proxy_patched:
        return
    _proxy_patched = True

    import httpx
    from qdrant_client.http.api_client import ApiClient
    from qdrant_client.http.exceptions import ResponseHandlingException

    _proxy_client = httpx.Client(proxy=proxy_url, timeout=60, verify=True)
    _original_send_inner = ApiClient.send_inner

    def _patched_send_inner(self, request):
        try:
            response = _proxy_client.send(request)
        except Exception as e:
            raise ResponseHandlingException(e)
        return response

    ApiClient.send_inner = _patched_send_inner
    logger.info("Qdrant ApiClient.send_inner 已 patch 使用代理: %s", proxy_url)


def _create_hybrid_collection(client: QdrantClient, collection_name: str):
    """创建混合向量集合：dense(1024) + sparse(BM25)。"""
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        },
    )


def _ensure_payload_indexes(client: QdrantClient, collection_name: str):
    """为常用过滤字段建 payload index（幂等）。

    Qdrant 要求被 filter 的 payload 字段必须先建索引（keyword/integer 等），
    否则 similarity_search(filter=...) 会返回 400:
        "Index required but not found for 'source' ..."
    """
    from qdrant_client.http.models import PayloadSchemaType
    try:
        schema = client.get_collection(collection_name).payload_schema or {}
    except Exception as e:
        logger.warning("读取 payload_schema 失败: %s", e)
        return

    existing = set(schema.keys() if hasattr(schema, "keys") else schema)
    # langchain_qdrant 把 metadata 存在 payload.metadata 嵌套对象下，
    # 过滤路径是 metadata.source / metadata.content_type / metadata.page
    targets = [
        ("metadata.source", PayloadSchemaType.KEYWORD),
        ("metadata.content_type", PayloadSchemaType.KEYWORD),
        ("metadata.page", PayloadSchemaType.INTEGER),
    ]
    for field, schema_type in targets:
        if field in existing:
            continue
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema_type,
            )
            logger.info("Created payload index: %s (%s)", field, schema_type)
        except Exception as e:
            # 已存在或创建失败都不影响主流程
            logger.debug("payload index %s: %s", field, e)


def ensure_collection(client: QdrantClient, collection_name: str):
    """如果集合不存在或 dense 维度不匹配，则（重新）创建混合集合，
    并确保过滤字段有 payload index。"""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        _create_hybrid_collection(client, collection_name)
        logger.info("Created Qdrant collection: %s (dense=1024)", collection_name)
        _ensure_payload_indexes(client, collection_name)
        return

    # 集合已存在 → 校验 dense 维度（embedding-3 为 1024 维）
    try:
        info = client.get_collection(collection_name)
        vecs = info.config.params.vectors
        if isinstance(vecs, dict):
            cur_size = vecs.get("dense").size
        else:
            cur_size = getattr(vecs, "size", None)
    except Exception as e:
        logger.warning("检查集合维度失败 (%s)，保留现有集合。", e)
        _ensure_payload_indexes(client, collection_name)
        return

    if cur_size is not None and cur_size != 1024:
        logger.warning(
            "维度不匹配 (现有 %s ≠ 1024)，删除重建集合: %s",
            cur_size, collection_name,
        )
        client.delete_collection(collection_name)
        _create_hybrid_collection(client, collection_name)
        logger.info("重建集合: %s (dense=1024)", collection_name)
    else:
        logger.info("Collection already exists: %s (dense=%s)", collection_name, cur_size)

    # 无论新建还是复用，都确保过滤字段有索引
    _ensure_payload_indexes(client, collection_name)


def get_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()
    ensure_collection(client, settings.qdrant_collection)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


# ── 文档提取 ───────────────────────────────────────────────────────────────────

def extract_with_docling(file_path: str) -> List[Document]:
    """从文件提取结构化内容，返回 LangChain Documents。

    优先用 Docling（富结构），未安装时按扩展名走轻量兜底：
    - PDF → PyMuPDF（已内置，按页提取文本+表格）
    - DOCX → zipfile 标准库解析 XML（零额外依赖）
    - TXT/MD → 直接读取
    """
    file_name = Path(file_path).stem
    file_hash = _file_hash(file_path)
    ext = Path(file_path).suffix.lower()

    # ── 优先：Docling（富结构：表格、版面）──────────────────────
    if DocumentConverter is not None:
        try:
            return _extract_with_docling_impl(file_path, file_name, file_hash)
        except Exception as e:
            logger.warning("Docling 解析失败 (%s)，回退到轻量解析器: %s", file_path, e)

    # ── 兜底：按扩展名分流 ──────────────────────────────────────
    if ext == ".pdf":
        return _extract_pdf_pymupdf(file_path, file_name, file_hash)
    if ext == ".docx":
        return _extract_docx_zipfile(file_path, file_name, file_hash)
    if ext in (".txt", ".md", ".markdown"):
        return _extract_text_file(file_path, file_name, file_hash)
    raise ValueError(f"不支持的文件类型: {ext}（支持 PDF/DOCX/TXT/MD）")


def _extract_with_docling_impl(file_path, file_name, file_hash) -> List[Document]:
    """Docling 实际实现：富结构（文本+表格）提取。"""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document
    documents = []

    # 文本分块（按页）
    full_md = doc.export_to_markdown()
    pages = full_md.split("<!-- page break -->")
    for i, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        if len(text) < 50:
            continue
        documents.append(Document(
            page_content=text,
            metadata={"source": file_name, "file_hash": file_hash,
                      "page": i, "content_type": "text"}
        ))
    # 表格
    for table in doc.tables:
        table_md = table.export_to_markdown()
        if table_md.strip():
            documents.append(Document(
                page_content=table_md,
                metadata={"source": file_name, "file_hash": file_hash,
                          "page": getattr(table, "page_no", 0), "content_type": "table"}
            ))
    logger.info("Docling 提取 %d 个文档块 from %s", len(documents), file_name)
    return documents


def _extract_pdf_pymupdf(file_path, file_name, file_hash) -> List[Document]:
    """用 PyMuPDF（已内置）解析 PDF：按页提取文本，表格转 markdown。"""
    import pymupdf
    documents = []
    doc = pymupdf.open(file_path)
    for i, page in enumerate(doc, start=1):
        # 文本
        text = page.get_text("text").strip()
        if len(text) >= 50:
            documents.append(Document(
                page_content=text,
                metadata={"source": file_name, "file_hash": file_hash,
                          "page": i, "content_type": "text"}
            ))
        # 表格（PyMuPDF 内置表格检测）
        try:
            tables = page.find_tables()
            for t in tables:
                # 转 markdown 字符串
                md = t.extract().to_markdown(index=False) if hasattr(t.extract(), "to_markdown") else None
                if md and md.strip():
                    documents.append(Document(
                        page_content=md,
                        metadata={"source": file_name, "file_hash": file_hash,
                                  "page": i, "content_type": "table"}
                    ))
        except Exception:
            pass  # 表格提取失败不影响文本
    doc.close()
    logger.info("PyMuPDF 提取 %d 个文档块 from %s", len(documents), file_name)
    return documents


def _extract_docx_zipfile(file_path, file_name, file_hash) -> List[Document]:
    """用标准库 zipfile 解析 DOCX：提取 word/document.xml 的纯文本（零额外依赖）。"""
    import zipfile
    from xml.etree import ElementTree as ET

    documents = []
    with zipfile.ZipFile(file_path) as z:
        xml = z.read("word/document.xml")
    # DOCX 命名空间
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    tree = ET.fromstring(xml)
    paragraphs = []
    for p in tree.iter(f"{{{ns['w']}}}p"):
        texts = [t.text for t in p.iter(f"{{{ns['w']}}}t") if t.text]
        para = "".join(texts).strip()
        if para:
            paragraphs.append(para)

    full_text = "\n\n".join(paragraphs)
    if len(full_text) >= 50:
        documents.append(Document(
            page_content=full_text,
            metadata={"source": file_name, "file_hash": file_hash,
                      "page": 1, "content_type": "text"}
        ))
    logger.info("zipfile 提取 %d 个文档块 from %s (DOCX)", len(documents), file_name)
    return documents


def _extract_text_file(file_path, file_name, file_hash) -> List[Document]:
    """直接读取 TXT/Markdown。"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if not text:
        return []
    return [Document(
        page_content=text,
        metadata={"source": file_name, "file_hash": file_hash,
                  "page": 1, "content_type": "text"}
    )]


def extract_from_text(text: str, source: str = "manual") -> List[Document]:
    """将原始文本包装为文档以进行导入。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )
    chunks = splitter.create_documents([text], metadatas=[{
        "source": source,
        "content_type": "text",
    }])
    return chunks


# ── 导入 ─────────────────────────────────────────────────────────────────────

def ingest_documents(documents: List[Document]) -> int:
    """
    去重、分块（如需要），并插入 Qdrant。
    同时增量构建 Neo4j 知识图谱（如果开启）。
    返回新增的向量数量。
    """
    if not documents:
        logger.warning("无文档可导入。")
        return 0

    # 按哈希去重
    seen_hashes = set()
    unique_docs = []
    for doc in documents:
        h = hashlib.md5(doc.page_content.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    logger.info("Ingesting %d unique documents (deduped from %d).",
                len(unique_docs), len(documents))

    # 分块大的文本块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunked = []
    for doc in unique_docs:
        if doc.metadata.get("content_type") == "text" and len(doc.page_content) > settings.chunk_size:
            sub_chunks = splitter.split_documents([doc])
            chunked.extend(sub_chunks)
        else:
            chunked.append(doc)

    vs = get_vector_store()
    ids = vs.add_documents(chunked)
    logger.info("Successfully ingested %d chunks.", len(ids))

    return len(ids)


def ingest_file(file_path: str) -> int:
    """完整管道：文件路径 → Qdrant。"""
    docs = extract_with_docling(file_path)
    return ingest_documents(docs)


def ingest_directory(dir_path: str, extensions: List[str] = None) -> int:
    """导入目录中所有受支持的文件。"""
    extensions = extensions or [".pdf", ".docx", ".txt", ".md"]
    total = 0
    for f in Path(dir_path).rglob("*"):
        if f.suffix.lower() in extensions:
            logger.info("Ingesting: %s", f)
            total += ingest_file(str(f))
    return total


# ── 辅助函数 ───────────────────────────────────────────────────────────────────

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()
