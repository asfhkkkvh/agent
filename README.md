# OmniRAG — 多智能体混合研究平台

基于 LangGraph 的多 Agent RAG 系统：监督者路由 → 混合检索（dense + BM25 稀疏 + cross-encoder 重排）→ 综合 → 自评审闭环。自带 FastAPI REST/SSE 接口、React 19 前端、MCP Server 与 RAGAS 4 维量化评估。

## 架构

```
                        ┌────────────┐
  用户查询 ─────────────►│ supervisor │  (监督者路由)
                        └─────┬──────┘
              rag_agent / web_agent / both
                        ┌─────┴──────┐
                        │  rag_node  │──► Qdrant 混合检索
                        │  web_node  │──► Tavily 网页搜索
                        │ both_node  │     (asyncio.gather 并行)
                        └─────┬──────┘
                              ▼
                        ┌────────────┐
                        │ synthesis  │  综合 RAG+Web 上下文生成答案
                        └─────┬──────┘
                              ▼
                        ┌────────────┐
                        │ critique   │  评审（忠实性/完整性/准确性/清晰度）
                        └─────┬──────┘
                   PASS / 达上限 ─────► 输出 final_answer
                   REVISE ────────────► 携带反馈回到 synthesis
```

检索链路（`app/rag/retriever.py`）：

```
Dense(智谱 embedding-3)  ┐
                         ├─ RRF 融合 ─► Cross-Encoder ─► top-N
Sparse(FastEmbed BM25)   ┘             (bge-reranker-base)
```

## 特性

- **监督者多 Agent 工作流**：LLM 路由到 知识库/网络/双路并行/直接综合，双路检索真正并行（`asyncio.gather`）
- **混合检索**：dense + sparse 倒数秩融合 + 本地 cross-encoder 重排，支持 LLM 提取元数据过滤器与多查询扩展
- **自评审闭环**：评审 Agent 的 REVISE 反馈传回综合 Agent 迭代修订，默认最多 5 轮，可通过 `MAX_ITERATIONS` 调整
- **对话记忆**：LangGraph SQLite checkpoint，同一 `thread_id` 延续上下文
- **接口完整**：FastAPI REST + SSE 流式事件（路由/检索/综合/评审逐步可见）、文件/文本导入、知识库统计、MCP Server
- **RAGAS 评估**：人工黄金集 + RAGAS 4 维 LLM-judge（faithfulness / answer_relevancy / context_precision / context_recall）
- **LangSmith 可观测**：全链路追踪
- **国内网络适配**：自动读取系统代理、Qdrant 走代理 patch、HF 镜像、多级文档解析降级

## 快速开始

前置依赖：Python 3.10+、Node 18+、一个智谱 GLM API Key（免费额度的 `glm-4-flash` / `embedding-3` 即可）、Qdrant Cloud 免费实例、Tavily 免费 Key。

```bash
# 1. 后端
cp .env.example .env          # 填入 ZHIPUAI_API_KEY / LANGCHAIN_API_KEY / QDRANT_* / TAVILY_API_KEY
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000

# 2. 导入文档（任一）
python scripts/ingest.py --file docs/你的文档.pdf
python scripts/ingest.py --dir ./my_docs
python scripts/ingest.py --text "要导入的文本" --label "我的资料"
# 或 curl:
curl -X POST -F "file=@你的文档.pdf" http://localhost:8000/api/ingest/file

# 3. 前端
cd frontend
npm install
npm run dev                   # 开发模式，自动代理 /api 到 8000

# 生产模式（前端构建后由 FastAPI 同源托管）
npm run build
```

## 环境变量

见 `.env.example`。核心项：

| 变量 | 说明 |
| --- | --- |
| `ZHIPUAI_API_KEY` | 智谱 GLM Key（LLM + embedding） |
| `ZHIPU_MODEL` | 默认 `glm-4-flash` |
| `ZHIPU_EMBEDDING_MODEL` | 默认 `embedding-3`（1024 维） |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant Cloud 向量库 |
| `TAVILY_API_KEY` | Tavily 网页搜索 |
| `LANGCHAIN_API_KEY` | LangSmith 追踪 |
| `MAX_ITERATIONS` | 评审闭环最大轮数（默认 5） |
| `RETRIEVAL_TOP_K` / `RERANKER_TOP_N` | 检索/重排数量 |
| `EVAL_GOLDEN_PATH` / `EVAL_REPORT_DIR` | 黄金集路径与评估报告目录 |

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/config` | 应用配置（不含密钥） |
| GET | `/api/stats` | 知识库统计（分块数 / 来源分布） |
| POST | `/api/query` | 同步查询 |
| POST | `/api/query/stream` | 流式查询（SSE，逐步推送 Agent 状态） |
| POST | `/api/ingest/file` | 上传文件导入 |
| POST | `/api/ingest/text` | 文本导入 |
| POST | `/api/evaluate` | RAGAS 评估（黄金集 + 4 维 LLM-judge） |

## 评估

评估基于**人工校验的黄金集**（`data/golden_set.json`），用 **RAGAS** 框架计算 4 个指标（judge LLM 用智谱 GLM）：

- **faithfulness**：答案关键论断是否被检索上下文支撑（无幻觉）
- **answer_relevancy**：答案是否切题、完整覆盖问题
- **context_precision**：检索上下文是否包含回答问题所需的信息
- **context_recall**：标准答案中的信息是否被检索到

```bash
# CLI 跑评估（需先导入文档并填写黄金集）
python -m app.rag.evaluation

# 或通过 API
curl -X POST http://localhost:8000/api/evaluate -H 'Content-Type: application/json' -d '{"k": 10}'
```

报告自动存档到 `data/eval_reports/eval_<时间戳>.json`，前端「质量评估」页可视化。

> 黄金集模板：`data/golden_set.json` 内置 12 条样本，请按你实际导入的知识库内容替换与扩充。`query` 必须能从库中检索到答案。
> 依赖说明：pyarrow 需 ≥17（旧版 `MonthDayNano` pickle bug 已修复）；judge 与生成同模型，指标用于相对度量迭代效果。

## 项目结构

```
app/
├── api.py               # FastAPI REST + SSE
├── config.py            # pydantic-settings 集中配置
├── graph/
│   └── workflow.py      # LangGraph 状态图（监督者/检索/综合/评审）
├── agents/              # rag_agent / web_agent / synthesis_agent / critique_agent
├── rag/
│   ├── retriever.py     # 混合检索（dense+sparse RRF + 重排）
│   ├── ingestion.py     # 文档解析→分块→向量化→入库
│   ├── reranker.py      # bge-reranker-base 重排
│   └── evaluation.py    # 黄金集 + RAGAS 4 维评估（faithfulness/relevancy/precision/recall）
└── mcp/server.py        # MCP Server（rag_search / web_search / full_query / evaluate）
frontend/                # React 19 + Vite + Tailwind + shadcn/ui
scripts/ingest.py        # CLI 导入工具
tests/                   # pytest 单元测试（路由/评审/检索格式化/黄金集…）
data/                    # 向量库 checkpoint、黄金集、评估报告
Dockerfile               # 多阶段构建（前端 + 后端）
```

## 重构说明（2026-09）

- **移除知识图谱（Neo4j）**：原 `kg_retrieve` 为 jieba 关键词 + 一跳邻居，非真正的图推理，且每个 chunk 一次 LLM 实体抽取成本高、收益低。移除后依赖（neo4j/jieba）与安全隐患（`aura_query` 的 `verify=False`）一并消除。
- **轻量评估 → RAGAS**：曾因旧版 pyarrow `MonthDayNano` pickle bug 与结果 NaN 弃用 RAGAS 改用自建轻量评估（recall@k + LLM-as-judge）；2026-09 升级 pyarrow ≥17 后 bug 已消除，换回 RAGAS 4 维标准评估（faithfulness / answer_relevancy / context_precision / context_recall）。
- **`MAX_ITERATIONS` 参数化**：评估不再修改全局 `settings`（原全局可变状态在并发下会互相踩），而是通过 `run_query(..., max_iterations=...)` 传入。

## 测试与质量

```bash
pip install -r requirements.txt
ruff check app tests
pytest -q
```

单元测试覆盖：路由判定、评审通过/修订判定、检索上下文格式化、导入去重、MCP 工具、黄金集加载、`max_iterations` 参数化等，不依赖真实 API。

## 已知限制

- LLM 免费额度有限（GLM-4-Flash 有 RPM 限制），双源并行时可能触发限流，重试即可
- `langchain-community` 的智谱 embedding 集成已标记弃用，官方 `langchain-zhipuai` 包在部分镜像源不可用，待其可用后迁移
- API 未加鉴权与限流，仅适合本地/内网演示
- Docker 部署时重排模型（bge-reranker-base）与文档解析依赖（poppler/tesseract）体积较大，首次拉取较慢

## License

MIT
