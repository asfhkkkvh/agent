# OmniRAG — 多智能体混合研究平台

一个把 **混合检索 RAG** 与 **多智能体编排** 结合的研究助手：监督者路由查询，RAG 知识库与实时网络双源检索（真正并行），综合 Agent 生成答案，评审 Agent 校验并**携带反馈迭代修订**，全程 LangSmith 可观测，并用 RAGAS 做量化评估。

> 技术定位：演示级 MVP，代码结构清晰、可自行部署，适合作为学习与面试作品项目。

## 特性

- **监督者路由**：LangGraph 状态机把查询路由到知识库 / 网络 / 双源并行
- **混合检索**：Dense（智谱 embedding）+ Sparse（BM25）双路向量，RRF 融合，Cross-Encoder（bge-reranker）重排，可选多查询扩展（Multi-Query）
- **知识图谱**：Neo4j 实体关系抽取 + 关键词多跳检索（中文用 jieba 分词），可选开关
- **真正的评估-修订闭环**：评审 Agent 的 REVISE 反馈会传回综合 Agent，逐条修订后再评审，直到 GOOD 或达到最大轮数
- **真并行**：RAG 与 Web 检索通过 asyncio 并发执行
- **对话记忆**：SQLite checkpoint 持久化，监督者与综合 Agent 都能看到最近对话
- **流式执行**：SSE 实时推送 路由 → 检索 → 综合 → 评审 的每一步
- **量化评估**：RAGAS 四指标（Faithfulness / Answer Relevancy / Context Precision / Recall），评测报告自动存档
- **双入口**：React + shadcn/ui 高级感前端（默认）、Streamlit 精简版（备用）
- **可观测**：LangSmith 全链路追踪

## 架构

```
React (shadcn/ui)  ──SSE──▶  FastAPI  ──▶  LangGraph 工作流
        ▲                        │              │
        │                        │              ▼
        └── /api/query/stream ◀──┘        监督者（路由）
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                          RAG Agent       Web Agent     双源并行
                       (混合检索+重排)   (Tavily)     (asyncio.gather)
                              └───────────────┼───────────────┘
                                              ▼
                                        综合 Agent
                                     （携带评审反馈修订）
                                              ▼
                                        评审 Agent
                                     GOOD? ──否──▶ 回到综合
                                              │是
                                              ▼
                                    最终答案 + 来源引用
```

## 技术栈

| 层级 | 技术 |
|---|---|
| LLM / Embeddings | 智谱 GLM-4-Flash、embedding-3 |
| 混合检索 | Qdrant（dense + sparse）、FastEmbed BM25、RRF 融合 |
| 重排 | BAAI/bge-reranker-base（本地） |
| 知识图谱 | Neo4j + jieba 分词 |
| 编排 | LangGraph（异步节点） |
| 记忆 | SQLite checkpoint |
| 可观测性 | LangSmith |
| 评估 | RAGAS |
| Web 搜索 | Tavily |
| 后端 | FastAPI + SSE |
| 前端 | React 19 + Vite + TypeScript + Tailwind CSS v4 + shadcn/ui |
| 部署 | Docker / docker-compose |

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
```

至少需要填写：`ZHIPUAI_API_KEY`、`LANGCHAIN_API_KEY`、`QDRANT_URL`、`QDRANT_API_KEY`、`TAVILY_API_KEY`。知识图谱相关的 `NEO4J_*` 与 `USE_KG_RETRIEVAL` 可选，默认关闭。

### 2. 安装后端依赖

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

PDF/DOCX 解析依赖 Docling，镜像源 403 时手动安装：`pip install docling`（不装也能用 TXT/MD/文本导入）。

### 3. 启动后端

```bash
uvicorn app.api:app --reload --port 8000
```

### 4. 启动前端（开发模式）

需要 Node.js 18+：

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173，自动代理 /api 到 8000
```

生产模式：`npm run build` 后，FastAPI 会自动托管 `frontend/dist`，直接访问 `http://localhost:8000` 即可。

### 5. 导入文档

```bash
python scripts/ingest.py --file docs/report.pdf
python scripts/ingest.py --dir ./my_docs
python scripts/ingest.py --text "要导入的文本" --label "我的资料"
```

也可以在界面的「知识库」页面上传文件或粘贴文本。

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
| POST | `/api/evaluate` | RAGAS 评估 |

## 测试与质量

```bash
pip install ruff pytest pytest-asyncio
ruff check app scripts tests
pytest tests -v
```

当前 23 个单元测试全部通过（重排、路由、评审判定、KG 分词、MCP 工具、检索多查询路径等），不依赖真实 API。

## 项目结构

```
app/
  graph/workflow.py       # LangGraph 异步工作流（路由/并行/评审闭环/流式事件）
  agents/                 # 监督者、RAG、Web、综合、评审
  rag/                    # 混合检索、重排、知识图谱、导入、RAGAS 评估
  mcp/server.py           # MCP 工具服务器
  api.py                  # FastAPI 接口 + 前端静态托管
  main.py                 # Streamlit 备用入口
frontend/                 # React + shadcn/ui 前端
tests/                    # 单元测试
data/eval_reports/        # RAGAS 评估报告（自动生成）
```

## 已知限制

- LLM 免费额度有限（GLM-4-Flash 15 RPM），流式查询在双源并行时可能触发限流，重试即可
- RAGAS 评估跑完整管道，样本多时耗时较长，建议先用 3-5 条样本
- API 未加鉴权与限流，仅适合本地/内网演示
- `langchain-community` 的智谱 embedding 集成已标记弃用，官方 `langchain-zhipuai` 包在部分镜像源还是空壳，待其可用后迁移

## License

MIT
