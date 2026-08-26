"""
OmniRAG — Streamlit 生产级 UI
────────────────────────────────────
简洁的生产级聊天界面，包含：
- 文件导入侧边栏
- 聊天历史
- 来源归因显示
- Agent 追踪可见性
- LangSmith 追踪链接

双入口说明：
- 交互式 UI：  streamlit run app/main.py     （本文件）
- REST API：   uvicorn app.api:app --reload   （见 app/api.py）
两条入口共用同一套核心函数（run_query / ingest_file 等）。
"""

import logging
import os
import sys
import tempfile
import time
import uuid

# 确保项目根目录在 sys.path 中，以便 `from app.graph.workflow import run_query` 能解析
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# .libs 目录装了所有第三方依赖（dotenv/langchain_community/langchain_qdrant/fastembed 等），
# 不在系统 site-packages 里，必须显式加入 sys.path 否则 import 会失败
_LIBS_DIR = os.path.join(_PROJECT_ROOT, ".libs")
if os.path.isdir(_LIBS_DIR) and _LIBS_DIR not in sys.path:
    sys.path.insert(0, _LIBS_DIR)

import streamlit as st
from dotenv import load_dotenv

# 加载 .env 到 os.environ（本地开发用；Streamlit Cloud 用 st.secrets 覆盖）
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# 从 Windows 注册表读取系统代理（Clash/V2Ray 等），Python 默认不读注册表代理
# 不设代理会导致直连 Qdrant Cloud（南美 AWS）被 GFW 干扰 → SSL EOF
if not os.environ.get("HTTPS_PROXY") and not os.environ.get("HTTP_PROXY"):
    try:
        import winreg
        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
            proxy_enable = winreg.QueryValueEx(key, "ProxyEnable")[0]
            if proxy_enable:
                proxy_server = winreg.QueryValueEx(key, "ProxyServer")[0]
                proxy_url = f"http://{proxy_server}"
                os.environ["HTTP_PROXY"] = proxy_url
                os.environ["HTTPS_PROXY"] = proxy_url
    except Exception:
        pass

# HuggingFace 模型下载走国内镜像，避免直连 huggingface.co 超时
# FastEmbed (Qdrant/bm25) 和 sentence_transformers (bge-reranker-base) 都从这里下载
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── 页面配置（必须是第一个 Streamlit 调用）────────────────────────────────────────
st.set_page_config(
    page_title="OmniRAG — Multi-Agent Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 设置环境变量（在导入前）— 本地用 .env；Streamlit Cloud 用 st.secrets
for _k in ["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT",
           "ZHIPUAI_API_KEY", "ZHIPU_MODEL", "ZHIPU_EMBEDDING_MODEL",
           "QDRANT_URL", "QDRANT_API_KEY", "TAVILY_API_KEY"]:
    try:
        _v = st.secrets[_k]
        if _v:
            os.environ[_k] = str(_v)
    except Exception:
        pass  # 无 st.secrets → 回退到 .env
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "omnirag-production")

from app.graph.workflow import run_query
from app.rag.ingestion import extract_from_text, ingest_documents

logging.basicConfig(level=logging.INFO)

# ── 自定义 CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E5EAA 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-rag    { background: #E8F5E9; color: #2E7D32; }
    .badge-web    { background: #E3F2FD; color: #1565C0; }
    .badge-both   { background: #F3E5F5; color: #6A1B9A; }
    .source-card {
        background: #F8F9FA;
        border-left: 4px solid #2E5EAA;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .critique-pass { color: #2E7D32; font-weight: 600; }
    .critique-fail { color: #C62828; font-weight: 600; }
    .metric-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── 会话状态 ──────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ── 标题 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem">🧠 OmniRAG</h1>
    <p style="margin:0.3rem 0 0 0;opacity:0.9">
        Multi-Agent Hybrid Research Platform &nbsp;|&nbsp;
        LangGraph · MCP · LangSmith · Hybrid RAG
    </p>
</div>
""", unsafe_allow_html=True)


# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 知识库")
    st.caption("上传文档以丰富 RAG 知识库")

    uploaded_files = st.file_uploader(
        "上传文件",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="支持 PDF、DOCX、TXT、Markdown",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.ingested_files:
                with st.spinner(f"正在处理 {uploaded_file.name}…"):
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=os.path.splitext(uploaded_file.name)[1],
                            delete=False
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        from app.rag.ingestion import ingest_file
                        count = ingest_file(tmp_path)
                        os.unlink(tmp_path)
                        st.session_state.ingested_files.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} — 已导入 {count} 个分块")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {e}")

    st.divider()

    # 快速文本导入
    with st.expander("📝 直接导入文本"):
        raw_text = st.text_area("粘贴文本内容", height=120, key="raw_text_input")
        text_label = st.text_input("标签/来源名称", value="手动输入", key="text_label_input")
        if st.button("导入文本", use_container_width=True):
            if raw_text.strip():
                docs = extract_from_text(raw_text, source=text_label)
                count = ingest_documents(docs)
                st.success(f"✅ 已导入 {count} 个分块")
                # 清空输入框
                st.session_state["raw_text_input"] = ""
                st.session_state["text_label_input"] = "手动输入"
                st.rerun()
            else:
                st.warning("请先输入一些文本。")

    st.divider()

    # 已导入文件列表
    if st.session_state.ingested_files:
        st.subheader("📚 已导入文件")
        for f in st.session_state.ingested_files:
            st.markdown(f"• {f}")

    st.divider()

    # 设置
    st.subheader("⚙️ 设置")
    show_context = st.toggle("显示检索上下文", value=False)
    show_critique = st.toggle("显示 Agent 评估", value=False)
    show_trace = st.toggle("显示 LangSmith 追踪", value=True)

    st.divider()

    # 统计
    st.subheader("📊 会话统计")
    col1, col2 = st.columns(2)
    col1.metric("查询数", st.session_state.total_queries)
    col2.metric("文件数", len(st.session_state.ingested_files))

    if st.button("🗑️ 清除聊天", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ── 聊天区域 ──────────────────────────────────────────────────────────────────
chat_col, info_col = st.columns([3, 1])

with chat_col:
    # 渲染历史记录
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

            # Agent 路由徽章
            route = turn.get("route", "")
            badge_class = {
                "rag_agent": "badge-rag",
                "web_agent": "badge-web",
                "both": "badge-both",
            }.get(route, "badge-both")
            st.markdown(
                f'<span class="agent-badge {badge_class}">🔀 {route or "both"}</span>'
                f'<span class="agent-badge" style="background:#FFF9C4;color:#F57F17">'
                f'⚡ {turn.get("iterations", 1)} 次迭代</span>',
                unsafe_allow_html=True,
            )

            if show_context and (turn.get("rag_context") or turn.get("web_context")):
                with st.expander("📖 检索上下文"):
                    if turn.get("rag_context"):
                        st.markdown("**📚 知识库：**")
                        st.markdown(turn["rag_context"])
                    if turn.get("web_context"):
                        st.markdown("**🌐 网络搜索：**")
                        st.markdown(turn["web_context"])

            if show_critique and turn.get("critique"):
                critique = turn["critique"]
                cls = "critique-pass" if "GOOD" in critique.upper() else "critique-fail"
                with st.expander("🔍 Agent 评估"):
                    st.markdown(f'<span class="{cls}">{critique}</span>', unsafe_allow_html=True)

    # 输入：优先消费「试试问」按钮触发的 _pending_query，否则读 chat_input
    query = st.session_state.pop("_pending_query", None) or st.chat_input(
        "输入你的问题 — 搜索你的文档 + 网络…"
    )
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🤔 思考中…（路由 → 检索 → 综合 → 评估）")

            start = time.time()
            try:
                result = run_query(query, thread_id=st.session_state.thread_id)
                elapsed = time.time() - start

                answer = result.get("final_answer") or "未生成答案。"
                placeholder.markdown(answer)

                # 路由徽章
                route = result.get("route", "")
                badge_class = {
                    "rag_agent": "badge-rag",
                    "web_agent": "badge-web",
                    "both": "badge-both",
                }.get(route, "badge-both")
                st.markdown(
                    f'<span class="agent-badge {badge_class}">🔀 {route or "both"}</span>'
                    f'<span class="agent-badge" style="background:#FFF9C4;color:#F57F17">'
                    f'⚡ {result.get("iterations", 1)} 次迭代</span>'
                    f'<span class="agent-badge" style="background:#ECEFF1;color:#455A64">'
                    f'⏱️ {elapsed:.1f}s</span>',
                    unsafe_allow_html=True,
                )

                if show_context:
                    with st.expander("📖 检索上下文"):
                        if result.get("rag_context"):
                            st.markdown("**📚 知识库：**")
                            st.markdown(result["rag_context"])
                        if result.get("web_context"):
                            st.markdown("**🌐 网络搜索：**")
                            st.markdown(result["web_context"])

                if show_critique and result.get("critique"):
                    critique = result["critique"]
                    cls = "critique-pass" if "GOOD" in critique.upper() else "critique-fail"
                    with st.expander("🔍 Agent 评估"):
                        st.markdown(f'<span class="{cls}">{critique}</span>', unsafe_allow_html=True)

                if show_trace:
                    trace_url = st.session_state.get("_langsmith_project_url")
                    if not trace_url:
                        try:
                            from langsmith import Client
                            _c = Client()
                            _p = _c.read_project(project_name=os.environ.get("LANGCHAIN_PROJECT", "omnirag-production"))
                            trace_url = f"https://smith.langchain.com/o/{_p.tenant_id}/projects/p/{_p.id}"
                            st.session_state["_langsmith_project_url"] = trace_url
                        except Exception:
                            trace_url = "https://smith.langchain.com/"
                    st.caption(f"📡 [查看 LangSmith 追踪]({trace_url})")

                # 保存到历史记录
                st.session_state.chat_history.append({
                    "query": query,
                    "answer": answer,
                    **result,
                })
                st.session_state.total_queries += 1

            except Exception as e:
                placeholder.error(f"❌ 错误: {e}")
                logging.exception("查询失败")


# ── 右侧信息面板 ──────────────────────────────────────────────────────────────
with info_col:
    st.subheader("🏗️ 架构")
    st.markdown("""
**处理流程：**
1. 🎯 **监督者** 路由查询
2. 📚 **RAG Agent** — 混合检索（Dense + BM25 → 重排序）
3. 🌐 **Web Agent** — Tavily 实时搜索
4. 🔗 **综合** — 合并双源信息
5. ✅ **评估** — 质量评估器（必要时循环）

**技术栈：**
- LangGraph（多 Agent）
- LangSmith（追踪）
- Qdrant（向量数据库）
- 智谱 GLM-4-Flash（LLM）
- BM25 + Dense 混合
- Cross-encoder 重排序
- MCP 服务器
- Streamlit UI
""")
    st.divider()
    st.subheader("💡 试试问：")
    sample_queries = [
        "上传报告中的关键发现是什么？",
        "总结最新的 AI 进展",
        "数据显示了哪些收入趋势？",
        "将我文档中的信息与当前新闻进行比较",
    ]
    for q in sample_queries:
        if st.button(q, key=q, use_container_width=True):
            # 直接把样例查询加入待执行队列，由主聊天区域消费
            st.session_state["_pending_query"] = q
            st.rerun()
