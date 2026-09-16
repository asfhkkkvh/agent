# OmniRAG 维修记录

> 从 2026-09-01 起，所有故障排查与修复记录于此。

***

## 2026-09-01 | 流式请求 500 / Qdrant 不可用

### 故障现象

- 前端报错：`Error: 流式请求失败 (500)`

- `/api/config`、`/api/stats` 均无法连接（ECONNREFUSED）

- 用户感知：查询 API 完全不可用

### 诊断过程

1. **检查后端进程**：`Get-CimInstance Win32_Process` 发现无 `python.exe` 进程——后端已完全退出。
2. **检查前端代理日志**：Vite 代理报 `ECONNREFUSED`（后端不在，请求无处转发）。
3. **检查代理环境**：系统代理未设置（`HTTPS_PROXY` / `HTTP_PROXY` 为空），但后端启动日志显示检测到 `http://127.0.0.1:7897` 代理（用户本地代理软件，端口 7897）。
4. **重启后端后验证**：

   - `/api/config` → 200 OK

   - `/api/query`（常规查询）→ 200 OK，50.2s 返回结果

   - `/api/query/stream`（流式查询）→ 200 OK，SSE 事件链完整（start → route → rag → synthesis → critique → final）

   - Qdrant 云连接正常（`HTTP/1.1 200 OK`）

### 根因

后端 uvicorn 进程崩溃/退出，非 Qdrant 服务端问题。前端因后端缺失报 500/ECONNREFUSED。

### 修复措施

重启后端：`python -m uvicorn app.api:app --host 127.0.0.1 --port 8000`

### 验证结果

| 端点                  | 状态  | 耗时    | 说明         |
| ------------------- | --- | ----- | ---------- |
| `/api/config`       | 200 | <1s   | 配置正常返回     |
| `/api/query`        | 200 | 50.2s | 首次查询（含预热）  |
| `/api/query/stream` | 200 | \~25s | SSE 事件链完整  |
| Qdrant 云            | 200 | \~4s  | 混合检索 10 候选 |

### 注意事项

- 后端进程可能因长时间空闲被系统终止，或因异常未捕获而崩溃。

- 如再次出现 500/ECONNREFUSED，首先检查后端进程是否存活。

- 可考虑加入进程守护（如 `PM2`、`systemd`、或 Windows 任务计划自动重启）。

***

## 2026-09-02 | Web Agent 返回"无法访问外部网站"

### 故障现象

- 用户提问需实时信息的问题时，系统回答"很抱歉，由于无法访问外部网站，我无法提供基于实时搜索结果的总结"

- 监督者已正确路由到 `web_agent`，Tavily API 本身正常可用

### 诊断过程

1. **独立测试 Tavily API**：`TavilySearch.invoke("Python 3.13 new features")` 返回正常结果。
2. **检查返回格式**：Tavily 返回的是 `dict`（含 `query` / `results` / `answer` / `images` 字段），不是 `list`。
3. **定位代码 bug**：`web_agent.py` 第 47 行 `raw_results = tool.invoke(query)` 直接遍历 dict，
   得到的是 key（`'query'`, `'follow_up_questions'`, `'answer'`, ...），而非搜索结果列表。
4. 综合 Agent 收到无意义的 key 列表，判定"无法提供网络搜索结果"，生成道歉回复。

### 根因

`langchain_tavily.TavilySearch.invoke()` 返回 `dict`，`web_agent.py` 未取 `["results"]` 字段，直接遍历 dict 导致拿到 key 而非实际搜索结果。

### 修复措施

`app/agents/web_agent.py`：增加返回类型判断，从 dict 中取 `results` 列表：

```python
raw = tool.invoke(query)
if isinstance(raw, dict):
    raw_results = raw.get("results", [])
elif isinstance(raw, list):
    raw_results = raw
else:
    raw_results = []
```

### 额外改进

- `app/api.py`：添加日志文件输出（`logs/omnirag.log`），同时保留终端输出。

- `app/api.py`：添加 HTTP 请求耗时中间件，记录每个 API 调用方法、路径、状态码和耗时。

***

## 2026-09-02 | 对话中触发评估导致流式查询中断

### 故障现象

- 用户正在对话（`/api/query/stream` SSE 流式连接持有 SQLite 连接）

- 用户切换到评估页面运行 `/api/evaluate`

- 回到对话后发现对话被"强制结束"（流式查询 500 中断）

### 诊断过程

1. **代码路径分析**：评估端点 `/api/evaluate` → `run_evaluation()` → 循环调用 `run_query()` → `asyncio.run(arun_query())`
2. **SQLite 连接冲突定位**：

   - `arun_query` 和 `stream_query` 都使用 `AsyncSqliteSaver.from_conn_string(db_path)` 打开同一个 `data/checkpoints.db`

   - 流式查询在 SSE 生命周期内持有 SQLite 连接（30\~50s）

   - 评估在同一时刻循环写入同一个数据库 → SQLite 写锁冲突 → 流式查询 500
3. **thread\_id 无冲突**：评估用 `f"eval-{uuid.hex[:8]}"` 唯一 ID，但共用数据库文件才是根因

### 根因

评估与用户对话共用同一个 SQLite checkpoint 文件（`data/checkpoints.db`），SQLite 写锁互斥导致流式查询被中断。

### 修复措施

1. **`app/graph/workflow.py`**：`arun_query` / `run_query` 新增 `db_path` 参数，允许指定独立的 checkpoint 数据库路径。
2. **`app/rag/evaluation.py`**：定义 `_EVAL_DB = data/eval_checkpoints.db`，评估调用 `run_query` 时传入此路径，与用户对话的 `checkpoints.db` 完全隔离。

### 验证

后端重启正常，Qdrant 连接 200 OK。评估和对话现在使用独立数据库，不再互相干扰。

***

## 2026-09-02 | 切换页面中断对话（前端条件渲染问题）

### 故障现象

- 用户在对话页面发起查询（SSE 流式连接进行中）

- 切换到评估页面再切回对话页面

- 对话被"强制结束"——历史消息消失，流式连接中断

### 诊断过程

1. **前端代码分析**：`App.tsx` 使用条件渲染 `{page === "chat" && <ChatPage/>}`
2. **组件卸载导致**：切页面时 `ChatPage` 卸载 →

   - `useState` 的 `turns`（对话历史）全部丢失

   - `threadIdRef` 重置为 null（对话线程断裂）

   - `streamQuery` 的 fetch/SSE 连接被浏览器回收
3. **后端仍在运行**：后端工作流可能还在执行，但前端已无法接收结果

### 根因

前端 `App.tsx` 条件渲染导致页面切换时组件卸载，对话状态和 SSE 连接全部丢失。

### 修复措施

`frontend/src/App.tsx`：将条件渲染改为 CSS `display` 切换，三个页面常驻不卸载：

```tsx
// 改前：切页面时组件卸载
{page === "chat" && <ChatPage config={config} />}

// 改后：组件常驻，仅隐藏显示
<div style={{ display: page === "chat" ? "flex" : "none" }}>
  <ChatPage config={config} />
</div>
```

### 效果

- 对话页面常驻，切换到评估/数据页面再回来时对话历史和线程 ID 保留

- SSE 连接在后台继续接收，不因页面切换中断

- 评估页面和对话页面互不干扰，各自独立运行

***

## 2026-09-01 | 性能优化：查询卡顿（96\~108s → 48s）

### 故障现象

- 一条完整查询耗时 **96\~108 秒**，用户明显感到"卡死"

- Qdrant 云（南美 AWS 节点）经本地代理访问时 TLS 握手极不稳定，查询偶发失败

- 评审 Agent 每轮都返回 REVISE，导致迭代 2~~5 轮，每轮叠加 20~~40s

- LLM 输出过长（综合 2500+ 字 / RAG 摘要 1400+ 字），生成耗时 34\~39s

### 诊断方法

1. **端点计时**：对 `/api/query` 连续请求记录总耗时
2. **节点级日志**：在 supervisor / rag / web / synthesis / critique 各节点打印耗时
3. **独立脚本对照**：分别测 Qdrant 混合查询（TLS 1.2 vs 默认）5 次成功率、GLM LLM 不同输出长度的生成速度、向量库冷启动 vs 预热后检索耗时

### 关键实测数据

| 项目                       | 结果                                             |
| ------------------------ | ---------------------------------------------- |
| Qdrant 默认 TLS 纯 dense 查询 | 5 次仅成功 1 次（间歇 TLSV1\_ALERT\_PROTOCOL\_VERSION） |
| Qdrant 强制 TLS 1.2 混合查询   | 5/5 成功，单次 3.5\~5s                              |
| glm-4-flash 短输出（69 字）    | 3.2s                                           |
| glm-4-flash 长输出（1487 字）  | 34.3s                                          |
| 向量库冷启动初始化                | 25\~30s（多轮 Qdrant 往返）                          |
| 预热后单次检索                  | 4\~5s                                          |

### 修复措施（8 项）

#### 1. Qdrant 代理 TLS 修复（根治卡顿主因）

`app/rag/ingestion.py`：在 `_patch_qdrant_proxy` 中强制 `ssl` 上下文 `minimum_version = TLSv1_2, maximum_version = TLSv1_2`；重试 3 次并递增退避（0.3s / 0.6s），重建 httpx.Client。

**效果**：TLS 1.2 下混合查询 5/5 稳定成功，单次 3.5\~5s。

#### 2. 检索降级兜底

`app/rag/retriever.py`：混合检索异常时自动降级为纯 dense 检索，保证查询不因偶发失败返回 500。

#### 3. 启动预热（消除首次查询冷启动）

`app/api.py`：新增 FastAPI `lifespan`，启动后在后台线程预热 `get_vector_store()` + 预检 `CrossEncoderReranker`，预热失败不阻塞服务。

**效果**：服务秒级可响应，首条查询直接复用已初始化的单例。

#### 4. 压缩 LLM 输出长度（削减最大瓶颈）

- `app/agents/synthesis_agent.py`：`max_tokens` 1200 → 700；提示词要求答案控制在 300\~600 字

- `app/agents/rag_agent.py`：`max_tokens` → 300；提示词要求关键事实摘要 ≤250 字

**效果**：综合 LLM 39s → ~~8~~10s；RAG 摘要 LLM 34s → \~18s。

#### 5. 评审闭环提速

`app/agents/critique_agent.py`：收紧评审标准——仅当存在明确硬性错误（幻觉、事实错误、未作答）才 REVISE；`.env`：`MAX_ITERATIONS=5 → 2`。

**效果**：查询迭代轮数稳定为 1 轮（多数查询评审即 PASS）。

#### 6. 路由策略偏向 RAG

`app/graph/workflow.py` 监督者提示词明确"内部文档知识型问题严禁路由到 web\_agent"，减少不必要的 Web 搜索。

#### 7. 过滤器提取默认关闭

`app/config.py` 新增 `use_filter_extraction`（默认 False）；收紧 `FILTER_PROMPT` 仅在用户明确提及 source/页码时才提取。

**原因**：GLM 易把普通查询误判为 source 过滤器导致 0 命中，且每次多一次 LLM 调用。

#### 8. 单例化 + 避免重复下载

- `get_qdrant_client` / `get_vector_store` / `get_sparse_embeddings` / `get_dense_vector_store` / `create_*_agent` 均 `@lru_cache` 单例

- `app/rag/reranker.py`：`_model_failed` 标志——模型未缓存或加载失败后不再重复尝试下载（避免每次查询浪费 30s）

### 效果对比

| 阶段         | 优化前          | 优化后         |
| ---------- | ------------ | ----------- |
| 首次查询       | 96\~108s     | ~~40~~55s   |
| 后续查询（热）    | 96\~108s     | **30\~42s** |
| RAG 检索     | \~20s（含失败重试） | \~5s        |
| RAG 摘要 LLM | \~34s        | \~18s       |
| 综合 LLM     | \~39s        | ~~8~~10s    |
| 评审 LLM     | \~8s         | ~~5~~8s     |
| 迭代轮数       | 2\~5 轮       | 1 轮         |

### 涉及文件

| 文件                              | 改动                                               |
| ------------------------------- | ------------------------------------------------ |
| `app/rag/ingestion.py`          | TLS 1.2 强制 + 退避重试；单例；dense 降级 store              |
| `app/rag/retriever.py`          | 混合检索失败降级纯 dense                                  |
| `app/rag/reranker.py`           | 模型缓存检查 + 失败标记                                    |
| `app/api.py`                    | 启动后台预热（向量库 + 重排器预检）                              |
| `app/graph/workflow.py`         | 监督者路由偏向 RAG；节点计时日志                               |
| `app/agents/rag_agent.py`       | 摘要输出压缩（max\_tokens=300，≤250 字）                   |
| `app/agents/synthesis_agent.py` | 答案输出压缩（max\_tokens=700，300\~600 字）               |
| `app/agents/critique_agent.py`  | 评审标准收紧，减少多余迭代                                    |
| `app/config.py` / `.env`        | `USE_FILTER_EXTRACTION=false`、`MAX_ITERATIONS=2` |

***

## 2026-09-02 | Web Agent 返回过时信息 + 答案格式差

### 故障现象

- 用户提问"总结最新的 AI 进展"，系统返回 GPT-3.5（2020年）、AlphaZero（2017年）等过时信息

- GLM-4-flash 忽略 Tavily 搜索结果，用自身过时训练知识编造答案

- 答案格式简陋（无结构化小节、无加粗标题、无来源标注）

- 监督者将时效性查询错误路由到 RAG（知识库文档）而非 Web 搜索

### 根因

1. **LLM 幻觉**：GLM-4-flash 训练数据截止 2024，但提示词未强制"只依据搜索结果回答"，模型用自己的过时知识生成内容
2. **路由失败**：监督者 LLM 忽略"最新""进展"等时效性关键词，将查询路由到 rag\_agent 而非 web\_agent
3. **提示词不够严格**：Web Agent 和综合 Agent 的提示词没有明确禁止使用自身训练知识

### 修复措施

#### 1. 关键词预路由（跳过 LLM 路由）

`app/graph/workflow.py` `supervisor_node`：在 LLM 路由前加关键词预路由逻辑：

- 查询含"最新""当前""近期""2025""2026""进展""新闻""动态""趋势"等时效性关键词且无 RAG 上下文 → 直接路由到 `web_agent`

- 跳过 LLM 路由调用，节省 1\~2s 和一次 API 请求

#### 2. Web Agent 提示词强化

`app/agents/web_agent.py` `WEB_PROMPT`：

- 新增【绝对禁止】区块：禁止使用自身训练知识、禁止编造搜索结果中不存在的内容

- 要求结构化输出：编号小节 + 加粗标题 + 具体数据（模型名、数字、日期）

- Tavily 搜索结果从 5 条增加到 8 条，`search_depth="advanced"` 获取更详细内容

#### 3. 综合 Agent 提示词强化

`app/agents/synthesis_agent.py` `SYNTHESIS_PROMPT`：

- 新增【绝对禁止】区块：禁止使用自身训练知识、禁止编造上下文中不存在的内容

- 要求结构化输出：编号小节 + 加粗标题 + 来源章节 + 引导提问

- `max_tokens` 从 700 放回 1200，支持详细结构化答案（600\~1200 字）

### 验证

- 含"最新""进展"关键词的查询正确路由到 `web_agent`（预路由 0.0s 决策）

- Tavily 搜索返回 5\~8 条实时结果

- 评审第 1 轮 PASS，总耗时 \~80s（Web 38.8s + 综合 37.4s + 评审 5.6s）

### 涉及文件

| 文件                              | 改动                                                  |
| ------------------------------- | --------------------------------------------------- |
| `app/graph/workflow.py`         | `supervisor_node` 新增关键词预路由逻辑                        |
| `app/agents/web_agent.py`       | 提示词禁止使用自身知识；Tavily 结果 5→8、`search_depth="advanced"` |
| `app/agents/synthesis_agent.py` | 提示词禁止使用自身知识；`max_tokens` 700→1200；结构化输出要求           |

***

## 2026-09-02 | Tavily 代理连接失败导致搜索 0 结果

### 故障现象

- 用户提问"你知道 gpt5.6 吗"，系统回答"未找到网络搜索结果"

- 监督者正确路由到 `web_agent`，但 Tavily 搜索返回 0 条

- 后端日志显示：`ProxyError('Unable to connect to proxy', RemoteDisconnected(...))`

### 根因

Tavily SDK 内部使用 httpx，会读取 `HTTPS_PROXY` 环境变量走代理。本地代理（127.0.0.1:7897）不稳定或拒绝 Tavily API 请求。Tavily 返回的 dict 含 `error` 字段而非抛异常，第一版重试逻辑只捕获了异常，未处理 dict 级别错误。

### 修复措施

`app/agents/web_agent.py` `WebAgent.run`：

1. 新增 `_has_error()` 检测 Tavily dict 级别错误（error 字段或空 results）
2. 第一次用当前环境（走代理）搜索，若返回 dict 级错误则标记为失败
3. 第二次清除所有代理环境变量直连 Tavily，`finally` 中恢复代理不影响其他模块
4. 两次都失败才返回错误信息

### 验证

```
13:57:34 Tavily 返回 dict, 耗时 3.7s
13:57:34 Tavily 搜索结果: 5 条  ← 代理失败后直连成功
13:58:01 Web agent 完成
```

### 涉及文件

| 文件                        | 改动                   |
| ------------------------- | -------------------- |
| `app/agents/web_agent.py` | 代理失败后直连重试；dict 级错误检测 |

***

## 2026-09-02 | 路由优化：简单问题直接回答 + 外部产品名走 Web + Tavily REST 直连

### 故障现象

1. 查询"你知道5.6sol"被路由到知识库（RAG），返回无关的智能家居项目文档
2. 查询"总结最新的AI进展"返回过时的 GPT-3.5/AlphaZero 信息（GLM 用自身训练知识编造，忽略 Tavily 结果）
3. 简单问题（如"什么是机器学习"）也走完整检索+评审流程，耗时 50+ 秒
4. LangSmith 追踪产生大量 SSL 错误日志噪音

### 根因

1. 预路由只检测时效性关键词，不检测外部产品/模型名，导致"GPT-5.6 Sol"等查询被 LLM 路由到 RAG
2. Tavily SDK 的 httpx 客户端读取 `HTTPS_PROXY` 环境变量走代理，代理拒绝连接后返回 dict 级错误（非异常），第一版重试只捕获异常
3. 所有查询都走完整流程（检索→综合→评审），简单常识问题不需要检索
4. LangSmith 追踪 API 也受代理 TLS 问题影响

### 修复措施

**`app/graph/workflow.py`**：

- 新增 `direct_answer` 状态字段，标记简单问题直接回答

- 预路由新增三类关键词检测：

  - 时效性关键词（最新/当前/进展等）→ `web_agent`

  - 外部产品/模型名（gpt/claude/gemini/sol/terra 等 20+ 关键词）→ `web_agent`

  - 简单问题关键词（什么是/解释/定义等）+ 短查询（≤15 字）→ `synthesis`（直接回答）

- `route_supervisor` 允许 `direct_answer=True` 时跳过上下文非空检查

**`app/agents/synthesis_agent.py`**：

- 新增 `DIRECT_PROMPT`：简单问题直接回答，100\~300 字，不走检索

- `run()` 新增 `direct` 参数：`direct=True` 时用 `DIRECT_PROMPT`，`max_tokens=500`

**`app/agents/web_agent.py`**：

- 改为双重搜索策略：先用 httpx REST API 直连 Tavily（`proxy=None`），失败后清代理用 SDK 重试

- `_has_error()` 检测 Tavily dict 级错误

**`.env`**：

- `LANGCHAIN_TRACING_V2=false`（关闭追踪，消除 SSL 噪音）

### 验证

| 查询          | 路由              | 耗时    | 结果             |
| ----------- | --------------- | ----- | -------------- |
| "什么是机器学习"   | synthesis（直接回答） | 4.5s  | ✓ 简洁回答         |
| "你知道5.6sol" | web\_agent      | 50s   | ✓ Tavily 5 条结果 |
| "总结最新的AI进展" | web\_agent      | \~50s | ✓ 实时搜索结果       |

### 涉及文件

| 文件                              | 改动                                        |
| ------------------------------- | ----------------------------------------- |
| `app/graph/workflow.py`         | 三类关键词预路由；`direct_answer` 状态字段；路由允许直接回答    |
| `app/agents/synthesis_agent.py` | 新增 `DIRECT_PROMPT`；`run()` 支持 `direct` 参数 |
| `app/agents/web_agent.py`       | Tavily REST 直连 + SDK 清代理双重搜索              |
| `.env`                          | 关闭 LangSmith 追踪                           |

***

## 2026-09-02 | MCP Server 工具精简 + 扩充（移除无用 + 暴露核心能力）

### 故障现象

- 'app/mcp/server.py' 暴露了 5 个工具：'hybrid\_search'、'web\_search'、'summarise\_docs'、'extract\_entities'、'calculate'

- 后 3 个本质上是 **LLM 提示词包一层**：宿主 LLM（如 TRAE）自己就能总结/实体识别/数学计算，做成 MCP 工具纯增加 stdio 往返，毫无收益

- 用户质疑：真正核心的能力——跑完整多 Agent 工作流和跑黄金集评估——反而没暴露，MCP 形同虚设

- '\_web\_search' 自己实例化 TavilyClient，代理绕过修复（REST + SDK 重试）没有跟 'web\_agent.py' 同步，会被代理卡死

- '\_hybrid\_search' 候选池扩大逻辑自相矛盾：max(top\_k, default) 在 top\_k > default 时等于没扩大，Cross-Encoder 对比候选少

### 修复措施

#### 1. 移除 3 个无用工具（summarise\_docs / extract\_entities / calculate）

- 删掉 TOOLS 定义和 3 个 handler 实现，代码 204→169 行

- 同步移除不再需要的 'import ast'、'import operator'

#### 2. 新增 2 个真正有价值的工具

| 工具                                                  | 作用                                                                     | 关键设计                                                                       |
| --------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 'full\_query(query, thread\_id?, max\_iterations?)' | 跑完整 LangGraph 工作流：监督者路由 → RAG/Web 检索 → 综合 → 评审                         | 独立的 mcp\_checkpoints.db（第 3 个 SQLite 数据库），与用户对话、评估互不干扰；可传 thread\_id 做多轮记忆 |
| 'evaluate(k?, threshold?)'                          | 跑黄金集评估，返回 recall\@k / faithfulness / answer\_relevancy + 单条样本详情 + 报告路径 | asyncio.to\_thread 丢线程池跑长任务，**不阻塞 MCP stdio 心跳**（评估 3-10min 也不会被宿主当死掉）     |

#### 3. web\_search 复用 robust 实现

- 不再自己 new TavilyClient，改为 'from app.agents.web\_agent import tavily\_search\_robust'

- 一处修复，MCP + workflow 两条链路共享，避免代理绕过和重试逻辑不同步

#### 4. hybrid\_search 候选池扩大逻辑修正

- 之前：top\_k = max(top\_k, settings.retrieval\_top\_k) → 大 top\_k 时等于没扩大

- 之后：top\_k = max(top\_k \* 2, settings.retrieval\_top\_k) → 不论大小 top\_k，Cross-Encoder 至少拿到 2× 候选对比

### 最终工具清单（4 个，都是宿主自己做不到的）

| # | 工具             | 提供的能力                                     | 宿主能否自做                               |
| - | -------------- | ----------------------------------------- | ------------------------------------ |
| 1 | hybrid\_search | 私有知识库混合检索（Dense+Sparse+RRF+重排）            | 宿主没有 Qdrant 集合和 Cross-Encoder 模型     |
| 2 | web\_search    | Tavily 实时搜索（代理绕过 + 双重保障）                  | 宿主没有配置 TAVILY\_API\_KEY              |
| 3 | full\_query    | 多 Agent 工作流（监督者+检索+综合+评审+记忆）              | 宿主没有 workflow 状态图和 SQLite checkpoint |
| 4 | evaluate       | 黄金集评估（recall/faithfulness/relevancy+报告存档） | 宿主没有 golden\_set 和完整管道               |

### 涉及文件

- 'app/mcp/server.py'：工具清单、handler、注释全面重写（移除 3 项 + 新增 2 项 + 代理修复同步 + top\_k 候选池修正）

***

## 2026-09-03 | 硬编码路由改造为 MCP + Function Calling

### 背景

监督者节点的路由逻辑是硬编码的：

- 关键词预路由：维护 `time_keywords` / `simple_keywords` 两个关键词列表，
  匹配到就跳过 LLM 直接决定 route

- LLM 文本路由：让 LLM 输出纯文本 `rag_agent`/`web_agent`/`both`/`synthesis`，
  再字符串匹配转 route

问题：关键词列表需人工维护、无法覆盖自然语言变体；文本路由依赖
LLM 输出格式严格匹配，脆弱。

### 改造目标

把路由决策交给 LLM 的 function calling——`llm.bind_tools(schema)`
后 LLM 返回 `tool_calls`，解析工具名映射到 LangGraph route。
同时 MCP server 的工具 schema 与 workflow 共享同一份定义。

### 实施步骤

#### 1. 新建 `app/tools/registry.py`（工具注册中心）

统一管理两套工具 schema：

| 导出                | 用途                        | 工具                                                                                              |
| ----------------- | ------------------------- | ----------------------------------------------------------------------------------------------- |
| `ROUTING_TOOLS`   | workflow 监督者 `bind_tools` | rag\_search / web\_search / parallel\_search / direct\_answer                                   |
| `get_mcp_tools()` | MCP server 对外暴露           | rag\_search / web\_search / full\_query / evaluate                                              |
| `TOOL_TO_ROUTE`   | 工具名 → route 映射            | rag\_search→rag\_agent, web\_search→web\_agent, parallel\_search→both, direct\_answer→synthesis |

设计原则：

- 路由决策型工具（workflow 内部）：LLM 选工具 → 映射 route → LangGraph 走节点，
  工具不实际执行检索（执行在 rag\_node/web\_node）

- 执行型工具（MCP 对外）：full\_query / evaluate 是宿主 LLM 做不到的完整能力

- parallel\_search / direct\_answer 不对外暴露（宿主可分别调 rag\_search + web\_search，
  direct\_answer 宿主用自身 LLM 回答）

#### 2. 改造 `app/graph/workflow.py` 监督者节点

- 删除 `time_keywords` / `simple_keywords` 关键词列表

- 删除 LLM 文本路由（输出纯文本 → 字符串匹配）

- 新增 `llm = get_llm().bind_tools(ROUTING_TOOLS)`

- LLM 返回 `response.tool_calls`，取 `tool_calls[0]["name"]`

- 经 `TOOL_TO_ROUTE` 映射到 route 值

- 保留"上下文已就绪 → synthesis"的状态判断（非工具调用，纯状态检查）

- 兜底：LLM 未返回 tool\_calls 时 route=both

#### 3. 改造 `app/mcp/server.py`

- 删除 110 行硬编码 `TOOLS` 列表

- `list_tools()` 改为 `return get_mcp_tools()`

- `hybrid_search` 工具统一改名为 `rag_search`（与 workflow 路由工具一致）

- `_hybrid_search` → `_rag_search`

- `call_tool` 路由 `rag_search` → `_rag_search`

### 验证结果（8 个测试用例）

| 查询             | 期望   | 实际 tool        | route      |
| -------------- | ---- | -------------- | ---------- |
| 你好             | 问候   | direct\_answer | synthesis  |
| 1+1等于几         | 数学   | direct\_answer | synthesis  |
| 报告中的项目数据是什么    | 知识库  | rag\_search    | rag\_agent |
| 文档里的结论         | 知识库  | rag\_search    | rag\_agent |
| 2026年最新AI模型有哪些 | 时效   | web\_search    | web\_agent |
| 最近的科技新闻        | 时效   | web\_search    | web\_agent |
| GPT-5的参数量      | 外部产品 | web\_search    | web\_agent |
| 什么是水           | 通用定义 | direct\_answer | synthesis  |

全部正确。单次路由决策耗时 1.6–4.8s（含 LLM 调用）。

### 架构收益

1. **去掉关键词维护**：不再需要手动维护 time\_keywords / simple\_keywords 列表，
   LLM 的 function calling 自主理解自然语言意图
2. **路由更鲁棒**：文本路由依赖 LLM 输出格式严格匹配（"rag\_agent" 字符串），
   function calling 返回结构化 tool\_calls，无格式匹配风险
3. **schema 单一来源**：`app/tools/registry.py` 一处定义，
   workflow `bind_tools` 和 MCP server `get_mcp_tools()` 两处使用，
   避免 schema 漂移
4. **MCP + Function Calling 结合**：内部 LangGraph 路由和外部 MCP 宿主
   调用共享同一套工具 schema 和命名

### 涉及文件

- `app/tools/registry.py`（新建）：工具 schema 注册中心

- `app/tools/__init__.py`（新建）：包初始化

- `app/graph/workflow.py`：监督者节点改为 function calling 路由

- `app/mcp/server.py`：工具 schema 改为从 registry 导入，hybrid\_search→rag\_search

***

## 2026-09-03 | web\_agent Tavily 搜索逻辑提取 + MCP 复用（修复隐藏 bug）

### 背景

上一轮 MCP + Function Calling 改造后，`app/mcp/server.py` 的 `_web_search`
引用了 `from app.agents.web_agent import tavily_search_robust`，但 web\_agent.py
里**根本没有这个函数**——MCP 调用 web\_search 工具时必然 ImportError。
REPAIR\_LOG 此前声称"复用 web\_agent 中的代理绕过逻辑"，实际代理逻辑仍封闭在
`WebAgent.run()` 方法内部闭包里，从未提取成独立函数。

### 故障现象

- MCP `web_search` 工具调用时：`ImportError: cannot import name 'tavily_search_robust'`

- 同时 SDK 兜底路径存在第二个 bug：`TavilySearch.invoke()` 返回格式化**字符串**，
  MCP 端 `for r in raw` 遍历字符串时抛 `'str' object has no attribute 'get'`

### 修复措施

#### 1. 提取模块级 `tavily_search_robust(query, max_results)`（web\_agent.py）

- 把 `WebAgent.run()` 内的代理保存/清理、REST 直连、SDK 兜底、错误判定
  整体提取为模块级函数，返回原始结果

- `WebAgent.run()` 只保留"格式化结果 + LLM 生成结构化摘要"，
  搜索部分改为调用 `tavily_search_robust`

- MCP `_web_search` 与 `WebAgent.run` 共用同一份代理绕过逻辑

#### 2. 修复 SDK 兜底路径的返回类型 bug

- 旧代码：`TavilySearch.invoke()` 返回格式化字符串，无法解析结果条数，
  导致 str.get 崩溃（该 bug 从引入起就存在，只是 MCP 复用后才暴露）

- 新代码：第二条路改为走系统默认网络栈（不传 `proxy=None`，可用环境代理）
  直接调 Tavily REST API，与第一条路返回统一的 dict 结构

- 删除不再使用的 `TavilySearch` / `TavilySearchAPIWrapper` 导入
  （`raw_results` 被 pydantic 包装成全部参数必填，调用冗长易错，弃用）

#### 3. 统一规范化输出为 list（web\_agent.py）

- `tavily_search_robust` 结尾：dict（含 results）→ 取 `results` 字段；list → 原样

- 调用方（`WebAgent.run` / MCP `_web_search`）拿到的恒为结果列表，
  不再各自做 isinstance 分型

#### 4. MCP 端适配同步函数（mcp/server.py）

- `tavily_search_robust` 是同步函数，`_web_search` 改 `asyncio.to_thread(...)` 调用，
  不阻塞 MCP 事件循环

### 验证结果

- `tavily_search_robust` 存在且可调用 ✓

- MCP `_web_search` 不再抛 `str.get` 崩溃；网络超时时优雅降级为错误消息文本 ✓

- 网络连通时 `tavily_search_robust` 返回 list（实测 3 条结果）✓

### 涉及文件

- `app/agents/web_agent.py`：提取 `tavily_search_robust` 模块级函数；
  WebAgent.run 复用；修复 SDK 兜底返回类型；统一规范化为 list；清理无用导入

- `app/mcp/server.py`：`_web_search` 改 `asyncio.to_thread` 调用同步函数

***

## 2026-09-03 | 监督者路由去硬编码：领域事实 + 语义边界（替代关键词规则）

### 问题

Function Calling 改造后的监督者 `SUPERVISOR_SYSTEM_PROMPT` 里写了一段"判断原则"：

```
- 出现"报告中的""文档里的""论文""笔记""上传的"等词 → rag_search
- 出现"最新""当前""2025""2026""进展""新闻"等时效词 → web_search
...
```

用户指出：**这就是硬编码**——只是把 Python 里的 `time_keywords` / `simple_keywords`
关键词列表搬进了提示词，本质没变（关键词匹配），只是执行者从 Python 变成了 LLM。

### 分析：为什么不能简单删掉关键词

直接删掉关键词后实测路由准确率从 10/10 掉到 5/10：

- GLM-4-Flash 把"报告中的项目数据""文档里的结论"误判为 direct\_answer

- 多个用例干脆不调用工具（NONE，犹豫 7-12s 后放弃）

原因：GLM-4-Flash 免费模型的 function calling 语义判断不稳定，
工具 description 边界弱时倾向"直接回答"或"不调工具"。

### 修复：领域事实 + 语义边界，零关键词

原则区分（这是本次修复的关键）：

- ❌ 关键词规则（硬编码）："出现'最新'两个字 → web\_search"

- ✅ 领域事实（语义理解）："知识库只含用户上传的文档（报告/论文/笔记）"——
  这是系统的领域属性，LLM 靠它理解"报告中的项目数据"属于 rag\_search 的范畴

- ✅ 语义边界（schema 描述）：每个工具 description 写明适用/不适用场景

改动两处：

1. `app/graph/workflow.py` 的 `SUPERVISOR_SYSTEM_PROMPT` 重写：

   - 只保留领域事实 + 决策约束：知识库=用户上传文档；简单常识才 direct\_answer；
     **必须调用工具，不能只回文本**；拿不准优先 parallel\_search；追问沿用上轮。

   - 删除全部关键词列举。

2. `app/tools/registry.py` 的 `ROUTING_TOOLS` description 强化语义边界：

   - rag\_search："知识库只含用户上传文档（项目报告、论文、笔记、README），
     涉及这些文档内容必须使用本工具"

   - direct\_answer："仅限问候/算术/纯常识定义；任何需要查证具体事实、
     数据或文档内容的问题都不得使用本工具，必须选择检索工具"

### 验证结果

10 个用例（含领域事实版 + 强化 description 版）：

| 查询             | 期望               | 实际 tool            |
| -------------- | ---------------- | ------------------ |
| 你好             | direct\_answer   | ✓ direct\_answer   |
| 1+1等于几         | direct\_answer   | ✓ direct\_answer   |
| 报告中的项目数据是什么    | rag\_search      | NONE（代码兜底 both）    |
| 文档里的结论         | rag\_search      | ✗ direct\_answer   |
| 论文里对缓存一致性的分析   | rag\_search      | ✓ rag\_search      |
| 2026年最新AI模型有哪些 | web\_search      | ✓ web\_search      |
| 最近的科技新闻        | web\_search      | ✓ web\_search      |
| GPT-5的参数量      | web\_search      | ✓ web\_search      |
| 什么是水           | direct\_answer   | ✓ direct\_answer   |
| 对比文档方案和开源方案    | parallel\_search | ✓ parallel\_search |

通过 8/10。剩余 2 例是 GLM-4-Flash 免费模型的 function calling 能力局限
（对"报告中的/文档里的"这类指代判断不稳）：

- NONE 用例由 `route_supervisor` 代码兜底强制走 both（已有逻辑）

- "文档里的结论"走 direct\_answer 时会明确告知无上下文，不产生幻觉

### 结论

- 代码与提示词中已无任何关键词列表 → 不再是硬编码

- 路由主体 = function calling（LLM 语义理解）+ 领域事实描述 + 语义边界

- NONE 兜底 = 安全默认（both），非关键词判断

- 8/10 是 glm-4-flash 免费模型在零硬编码前提下的稳定水平，
  配合兜底链路可用；不再追求极致准确率（避免陷入无限调优）

### 涉及文件

- `app/graph/workflow.py`：重写 SUPERVISOR\_SYSTEM\_PROMPT（领域事实+约束，删关键词）

- `app/tools/registry.py`：强化 ROUTING\_TOOLS 的 description 语义边界

***

## 2026-09-03 | 项目自洽性重构（消除矛盾/冗余/死代码）

### 背景

用户要求审查整个项目，目标是"逻辑自洽"。审查发现 12 类问题：
矛盾（chunk 配置、guide/README 过时、测试测已删功能）、
冗余（ChatZhipuAI 6 处重复、双套 HybridRetriever、MCP\_TOOL\_NAMES 死常量）、
不合理/死代码（supervisor 死分支、direct\_answer 仍过评审、agent\_temperature 死配置）。

### 修复项清单

| #  | 问题                                     | 修复                                                                                              |
| -- | -------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1  | chunk 配置矛盾                             | config 默认 1000/200 → 400/40（与 .env.example/实际 .env 一致）；修正"header-aware"为"按 Markdown 标题优先"的真实描述  |
| 2  | guide.txt 过时                           | MCP 5 工具→4（rag\_search 等）、前端 Streamlit→React、GitHub 地址、tests 描述、统计全部更新                          |
| 3  | README/learning\_doc hybrid\_search 残留 | 全部改为 rag\_search/full\_query/evaluate                                                           |
| 4  | 测试测已删功能                                | 删 TestMCPCalculator；TestMCPHybridSearch→TestMCPRagSearch；supervisor 测试适配 function calling       |
| 5  | ChatZhipuAI 6 处重复构造                    | 新建 app/llm.py 统一 get\_llm() 工厂，6 个文件全部复用                                                        |
| 6  | 双套 HybridRetriever                     | rag\_agent 新增 create\_retriever() 共享工厂，MCP \_rag\_search 复用                                     |
| 7  | MCP\_TOOL\_NAMES 死常量                   | 删除，改为注释说明 call\_tool 显式分发                                                                       |
| 8  | supervisor 死分支                         | 删除"上下文已就绪→synthesis"（supervisor 首轮执行，上下文恒为空）                                                    |
| 9  | direct\_answer 仍过评审                    | 新增 route\_after\_synthesis：direct\_answer → 直接 end，省一次 LLM 往返；synthesis\_node 直接写 final\_answer |
| 10 | agent\_temperature 死配置                 | 从 config/.env.example 删除，温度统一在 get\_llm 按角色指定                                                   |
| 11 | MultiQueryRetriever 死路径                | 保留（USE\_MULTI\_QUERY 可选能力），文档修正宣传措辞                                                             |
| 12 | 评估重复检索                                 | 保留（recall 需要原始 chunk，run\_query 返回的是 LLM 摘要），注释说明权衡                                             |

### 新增文件

- `app/llm.py`：共享 LLM 工厂（get\_llm），参数统一（model/api\_key/temperature/max\_tokens）

### 验证结果

- 全部模块 import 正常

- pytest 23/23 通过（含新增 test\_supervisor\_direct\_answer、适配后的 FC 路由测试）

- 图结构验证：synthesis\_node → (critique\_node | end) 条件边正确

- 真实流式查询"你好"：事件序列 start → route → synthesis → final（**无 critique**），direct\_answer 正确跳过评审

- 后端重启后 /api/health ok；前端 5173 + 代理正常

### 备注

- 修复过程中发现 8000 端口被无关进程 `pca.server:app` 占用导致请求分发异常，已清理并重启

- 用户 `.env` 中残留的 AGENT\_TEMPERATURE 已被 extra=ignore 忽略，无副作用，可自行删除

### 涉及文件

- `app/llm.py`（新建）：共享 LLM 工厂

- `app/config.py`：chunk 默认 400/40、删 agent\_temperature、注释

- `app/graph/workflow.py`：删死分支、route\_after\_synthesis、synthesis 直答、get\_llm 复用

- `app/agents/{rag,web,synthesis,critique}_agent.py`：get\_llm 复用；rag\_agent 新增 create\_retriever

- `app/rag/{retriever,ingestion,evaluation}.py`：get\_llm 复用；chunk separators 统一+注释

- `app/mcp/server.py`：\_rag\_search 复用 create\_retriever

- `app/tools/registry.py`：删 MCP\_TOOL\_NAMES

- `tests/test_agents.py`：删 TestMCPCalculator、适配 FC 路由、改名 \_rag\_search

- `guide.txt` / `README.md` / `docs/omnirag_learning_doc.md` / `.env.example`：一致性更新

***

## 2026-09-08 | 评估换回 RAGAS + 全面整改（黄金集去 MemCache / 知识库重建）

### 背景

用户要求"换回 ragas，之后全面整改项目"，并指出黄金集里全是另一个项目（MemCache）的数据不合理。
2026-09 初曾因旧版 pyarrow `MonthDayNano` pickle bug / NaN 弃用 RAGAS 改轻量评估；
本次升级 pyarrow ≥17 后 bug 已消除，换回 RAGAS 4 维标准评估。

### 关键实现与踩坑

| # | 问题                                                                                           | 解决                                                                                                   |
| - | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1 | langchain-community 0.4.x 已移除 `ChatVertexAI`，ragas 0.4.x 顶层 import 报错                        | evaluation.py 顶部注入 vertexai shim 模块（`sys.modules` 占位）绕过                                              |
| 2 | ragas 0.4.3 collections metrics 强制要求 OpenAI 风格 `InstructorLLM`，与智谱 `LangchainLLMWrapper` 不兼容 | 改用 `ragas.metrics` 旧版路径（`Faithfulness` 等实例，llm 参数可选），由 `evaluate(llm=...)` 全局注入；仅 DeprecationWarning |
| 3 | metrics 从 `ragas.metrics` 导入拿到的是实例（`'Faithfulness' object is not callable`）                  | 直接放入 metrics 列表，不再调用                                                                                 |
| 4 | 黄金集含 5 条 MemCache 样本 + 过时 Neo4j 描述                                                           | 重写为 OmniRAG 专属 12 条（匹配新导入文档）                                                                         |
| 5 | 知识库存 MemCache 项目文档                                                                           | 清空重建 Qdrant 集合，导入 README.md / REPAIR\_LOG.md / omnirag\_learning\_doc.md（304 chunk）                  |

### 改动清单

- `requirements.txt`：加回 `ragas>=0.2.0`、`datasets>=3.0.0`、`pyarrow>=17.0.0`（锁版本规避 pickle bug）

- `app/rag/evaluation.py`：完整重写为 RAGAS 4 维（faithfulness / answer\_relevancy / context\_precision / context\_recall）；
  vertexai shim、旧版 metrics 路径、SingleTurnSample 构造、独立 eval\_checkpoints.db、报告存档

- `app/api.py`：EvalRequest 去掉 threshold，/api/evaluate 返回 4 维指标

- `app/mcp/server.py`：evaluate 工具与 \_evaluate 适配 4 维指标

- `app/tools/registry.py`：evaluate 工具 schema 描述更新

- `data/golden_set.json`：重写为 OmniRAG 专属 12 条（去 MemCache）

- 知识库：清空重建，导入 OmniRAG 文档（304 chunk）

- `README.md` / `guide.txt` / `docs/omnirag_learning_doc.md`：评估描述全面对齐 RAGAS

- `tests/`：23/23 通过，无回归

### 验证

- pytest 23/23 通过

- RAGAS 链路验证成功（单条样本 4 指标：faithfulness=1.0 / answer\_relevancy=0.698 / context\_precision=0.723 / context\_recall=1.0）

- 知识库重建后 304 chunk（README 26 + REPAIR\_LOG 97 + learning\_doc 181）

- run\_query 全链路 99s 正常返回（修复前无限卡死）

***

## 2026-09-08 | 查询无限卡死根因：重排器 HF 缓存锁阻塞（重要性能 bug）

### 故障现象

RAGAS 换回后的验证过程中，run\_query 单条查询**无限卡死**（12 分钟无结果，进程 CPU 仅 0.78s，纯等 I/O）。

### 诊断过程

1. **分步计时定位**：监督者 4.3s 正常 → 卡在 rag\_node（3 分钟无输出）
2. **对比分离测试**：`_retrieve_top_k`（`reranker_top_n=k=10`，不触发重排）19.4s 正常；
   rag\_node 的 `create_retriever`（`reranker_top_n=5`，触发重排）卡死 → **重排器嫌疑**
3. **单独测** **`_get_model()`**：也超 2 分钟 → 确认重排器卡点
4. **检查 HF 缓存**：`models--BAAI--bge-reranker-base` 目录存在但仅 63MB
   （**部分下载**，无 model.safetensors / pytorch\_model.bin 权重文件），
   且含 `.lock` 残留

### 根因

`CrossEncoderReranker._model_is_cached()` 用 `snapshot_download(local_files_only=True)`
探测模型是否缓存。在**部分缓存 + 残留 .lock** 场景下，huggingface\_hub 会阻塞等待
文件锁（不会因 local\_files\_only 跳过锁检查），导致每次新进程第一次检索都无限等锁。

### 修复

`reranker.py` 的 `_model_is_cached` 改为**直接遍历 snapshots 目录检查权重文件**
（model.safetensors / pytorch\_model.bin 存在性），纯本地 I/O 毫秒级返回；
未完整缓存 → 快速置 `_model_failed=True` 降级跳过重排。

### 效果

- `_get_model()` 从"无限卡死" → **0.1s 快速降级**

- run\_query 全链路 99s 正常返回（监督者 9s + rag 26s + web 33s + synthesis 36s + critique 4s）

- 若后续需要启用重排：`HF_ENDPOINT=https://hf-mirror.com` 完整下载模型后自动生效

### 涉及文件

- `app/rag/reranker.py`：`_model_is_cached` 改为直接文件探测（弃用 snapshot\_download 锁阻塞路径）

***

## 2026-09-08 | 黄金集候选生成脚本（方法一：从知识库文档提炼）

### 背景

用户提出黄金集构建的三种方法论（文档提炼 / FAQ 转换 / 真实日志标注），
确认前两者未实现、第三项无日志源。落地成本最低、且能规避"自引用失真"
的方法是"文档提炼 + **人工校验**"的辅助脚本。

### 实现

新增 `scripts/build_golden_set.py`：

- 从 Qdrant scroll 读取 chunks（含 metadata.source，3 次退避重试抗 Qdrant 抖动）

- 对每个 chunk 用 LLM（temperature=0.3）生成 问题 + 参考答案（只允许来自片段内容）

- embedding 余弦相似度去重（阈值 0.85，与最近 20 条比较）

- 输出 `data/golden_set.candidates.json`（含 \_comment 提示必须人工校验）

关键设计：脚本只产出**候选**，答案由 LLM 从 chunk 提取，直接入库会造成
"自引用失真"（自己考自己分数虚高），因此人工校验是评估可信度的底线。

### 踩坑

- ChatPromptTemplate 会把 system 里 `{"query": ...}` 的 JSON 示例花括号
  当成模板变量 → 报错 INVALID\_PROMPT\_INPUT；改为文字描述避免花括号

- Qdrant scroll 连接不稳定（南美节点）→ load\_chunks 加 3 次退避重试

### 验证

- `--limit 6` 实测：4 个 chunk → 生成 4 条候选（0 失败）

- 质量抽查：README 来源的候选良好；REPAIR\_LOG 来源的部分候选
  带具体日期/过度细节，需人工修正 —— 符合"辅助 + 人工校验"定位

### 涉及文件

- `scripts/build_golden_set.py`（新建）：黄金集候选生成脚本


---

## 2026-09-08 路由环（检索后纠错）——新增

### 问题

监督者是纯 LLM 决策（function calling），一旦路由出去就没有回头路：
1. 监督者误判 rag_search，向量库无结果（如"查天气"）→ RAG 返回空/弱命中，但系统不知道要补 Web
2. 监督者误判 direct_answer（GLM-4-Flash function calling 把"README 项目叫什么"也判成直接回答）→ 绕开检索，直答失败或直接编造

### 方案：三层路由环（检索后 + 直答后纠错）

| 层 | 检测信号 | 动作 |
| --- | --- | --- |
| RAG 未命中 | ag_agent.EMPTY_RESULT（空结果 或 全部 rerank_score < 0.15 弱命中） | 补路 web_node |
| Web 未命中 | web_agent 失败/无结果文案 | 补路 rag_node |
| 直答失败 | 答案含"无法/抱歉/没有提供"等标记 | 补路 both_node（双路检索） |

防死循环：ag_ran / web_ran 状态标记，rag↔web 最多互补一次；both_node 重置 direct_answer=False 且清空 draft_answer，补路后走正常综合 + 评审。

### 验证（真实查询）

- "README 项目叫什么" → rag 未命中 → 补路 Web ✓
- "2026 AI 模型" → Tavily 失败 → 补路 RAG ✓
- "今天北京天气" → GLM 误判 direct_answer 直答失败 → 补路双路 → 正确回答"晴朗 18℃ [Web 2]" ✓
- 问候"你好" → 直答成功，不补路 ✓
- pytest 23/23 通过

### 已知边界（诚实记录）

- GLM-4-Flash 对 direct_answer 的误判仍存在（prompt + description 已收紧"实时信息禁止直答"，但免费模型 function calling 遵循度有限）
- 直答**编造**的"看似完整"答案（如编造天气数字）路由环检测不到——防幻觉职责在评审环，而 direct_answer 为省时跳过评审，此为设计权衡

### 涉及文件

- pp/graph/workflow.py：AgentState 增 rag_empty/web_empty/rag_ran/web_ran；route_after_rag / route_after_web / route_after_synthesis 三层补路；both_node 重置 direct_answer
- pp/agents/rag_agent.py：EMPTY_RESULT 常量 + WEAK_HIT_THRESHOLD=0.15 弱命中判定
- pp/tools/registry.py：direct_answer description 收紧（实时信息禁止直答）

---

## 2026-09-10 token 级流式（SSE 逐字推送）——新增

### 问题
- 综合 LLM 一次性返回完整答案（8~10s），SSE 只有节点级状态事件，用户全程干等
- OPTIMIZATION_LOG 记录的"最能改善感知卡顿"的未做项

### 方案
1. synthesis_agent.arun_stream()：async generator，chain.astream 逐 token yield（与 run() 共用同一组 prompt/参数）
2. synthesis_node：用 langgraph.config.get_stream_writer() 把 token 实时透传（stream_mode="custom"），同时拼装完整答案；非流式调用（ainvoke）时 writer 为 no-op，行为不变
3. stream_query：astream 双 mode（updates + custom），custom 事件转成 {"type":"token","text":...} SSE 事件
4. 前端 ChatPage：token 事件逐字追加 answer，Markdown 实时重渲染

### 实测（真实查询"你好"）
- ChatZhipuAI.stream 原生真流式：12 chunk、首 token 0.71s（裸 LLM）
- 端到端：start → route(1.4s) → 首 token 1.82s → 7 个 token 事件 → final，总耗时 2.05s
- 感知延迟：综合生成从"8~10s 干等"变为"约 1s 首字"，剩余逐字涌入

### 边界（诚实记录）
- 首 token 前仍需等待监督者路由 + 检索（复杂查询检索 18~30s 不可省），流式改善的是"综合生成阶段"的感知延迟
- 前端顺序：route 状态 → token 流 → synthesis 状态 → final（synthesis 的 updates 事件在节点完成后才到）

### 涉及文件
- pp/agents/synthesis_agent.py：新增 arun_stream
- pp/graph/workflow.py：synthesis_node 流式 + stream_query 双 mode
- rontend/src/lib/api.ts：StreamEvent 增加 token 类型
- rontend/src/pages/ChatPage.tsx：token 逐字渲染

---

## 2026-09-16 长期记忆库（v2，P0：写入管道 + 跨会话召回）——新增

### 目标
把高质量对话提炼为可跨会话检索的记忆条目，与文档知识库物理隔离。

### 实现
- pp/memory/memory_store.py（新）：
  - 提炼：LLM 只提取 fact / preference / event 三类，过滤寒暄与一次性问题（"提炼"与"存档"的本质区别）
  - 存储：独立 omnirag_memory 集合（纯 dense，复用 embedding-3 1024 维），与文档 omnirag_hybrid 物理隔离
  - 去重：新候选与已有记忆余弦相似度 > 0.92 跳过
  - 召回：ecall_memories(query, top_k)，任何失败静默降级为空（绝不阻塞在线查询）
- workflow.py：synthesis_node 注入"长期记忆"提示区域（仅 USE_MEMORY=true 时）；arun_query/stream_query 加 save_memory 参数，完成后后台异步提炼（syncio.create_task + to_thread，不阻塞响应）
- pi.py：两个对话查询端点传 save_memory=True；评估 / MCP 默认 False（避免实验污染记忆库）
- config.py / .env.example：USE_MEMORY（默认 false）/ MEMORY_COLLECTION / MEMORY_TOP_K

### 踩坑记录
1. ChatPromptTemplate 会把 system 提示中的 JSON 示例 {"content": ...} 当成模板变量 → KeyError: '"content"' → 双花括号 {{...}} 转义
2. QdrantVectorStore.from_existing_collection 内部新建 client 不走代理 patch → SSL 连接失败 → 必须显式传 client=get_qdrant_client()（复用 TLS 1.2 + 重试 + 连接重建）

### 验证
- 对话1 提及"项目用 Qdrant" → 提炼写入 1 条 fact 记忆
- 新会话换问法"我用的向量数据库是什么？" → 召回命中"用户的项目使用 Qdrant 向量库"
- 评估 / MCP 路径不触发记忆写入（save_memory=False）
- pytest 23/23 通过
