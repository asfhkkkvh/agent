# OmniRAG 性能优化记录

> 目标：解决"项目卡顿"问题（查询响应过慢），并在优化中保持功能稳定。
> 记录每次定位、改动与实测数据，供后续开发与简历面试引用。

---

## 1. 背景与问题定义

多智能体混合 RAG 平台，一条完整查询的耗时曾高达 **96~108 秒**（远超可用阈值），
前端虽有 SSE 流式状态提示，但用户仍需长时间等待最终答案。

瓶颈排查聚焦两个方向：

- **网络链路**：Qdrant 云（南美 AWS 节点）经本地代理访问时 TLS 握手极不稳定，
  查询偶发失败，重试会浪费 20s+。
- **LLM 生成**：智谱 GLM-4-flash 生成速度实测约 **35~45 字符/秒**，
  管线中 3 个串行 LLM 调用（RAG 提取 → 综合 → 评审）的输出长度直接决定总耗时。

---

## 2. 诊断方法

1. **端点计时**：对 `/api/query` 连续请求并记录总耗时。
2. **节点级日志**：在 `supervisor / rag / web / synthesis / critique` 各节点打印耗时。
3. **Agent 级日志**：在 `rag_agent / synthesis_agent / critique_agent` 的检索与 LLM 调用处分别计时。
4. **独立脚本对照**：隔离变量，分别测：
   - Qdrant 混合查询（TLS 1.2 vs 默认）5 次成功率与耗时
   - GLM LLM 不同输出长度的生成速度
   - 向量库冷启动 vs 预热后的检索耗时

### 关键实测数据

| 项目 | 结果 |
| --- | --- |
| Qdrant 默认 TLS 纯 dense 查询 | 5 次仅成功 1 次（间歇 TLSV1_ALERT_PROTOCOL_VERSION） |
| Qdrant 强制 TLS 1.2 混合查询 | 5/5 成功，单次 3.5~5s |
| glm-4-flash 短输出（69 字） | 3.2s |
| glm-4-flash 长输出（1487 字） | 34.3s |
| 向量库冷启动初始化 | 25~30s（多轮 Qdrant 往返） |
| 预热后单次检索 | 4~5s |

---

## 3. 已实施的优化

### 3.1 网络链路：Qdrant 代理 TLS 修复（根治卡顿主因）

**问题**：Qdrant 云经本地代理时，默认 TLS 协商（1.3）被代理/服务器间歇性拒绝，
报 `[SSL: TLSV1_ALERT_PROTOCOL_VERSION]`，纯 dense 查询实测 5 次仅成功 1 次。
失败后每次重试浪费 ~20s。

**修复**（`app/rag/ingestion.py`）：
- 在 `_patch_qdrant_proxy` 中强制 `ssl` 上下文 `minimum_version = TLSv1_2, maximum_version = TLSv1_2`。
- 重试逻辑保留 3 次，并在重建连接之间加入递增退避（0.3s / 0.6s）。

**效果**：TLS 1.2 下混合查询 **5/5 稳定成功**，单次 3.5~5s。

### 3.2 检索降级兜底（稳定性）

**问题**：混合检索（prefetch dense+sparse）在南美云/代理下仍可能偶发失败。

**修复**（`app/rag/retriever.py`）：混合检索异常时自动降级为纯 dense 检索，
保证查询始终可用，不因偶发失败返回 500。

### 3.3 启动预热（消除首次查询冷启动）

**问题**：向量库首次初始化约 25~30s（含 Qdrant 云多次往返、嵌入器加载），
若放到首条查询才触发，用户第一问必然"卡死"。

**修复**（`app/api.py`）：
- 新增 FastAPI `lifespan`，启动后在**后台线程**预热：
  - `get_vector_store()` 初始化 Qdrant 客户端 + 集合校验 + dense/sparse 嵌入器；
  - 预检 `CrossEncoderReranker`，未缓存则提前标记失败，避免首条查询额外等待 ~8s 的模型检查。
- 预热失败不阻塞服务启动（查询时懒加载兜底）。

**效果**：服务秒级可响应，首条查询直接复用已初始化的单例。

### 3.4 压缩 LLM 输出长度（削减最大瓶颈）

**问题**：glm-4-flash 生成约 35~45 字符/s，输出越长耗时越长；
综合回答一度生成 2500+ 字（39s），RAG 摘要 1400+ 字（34s）。

**修复**：
- `app/agents/synthesis_agent.py`：`max_tokens` 1200 → 700；
  提示词要求答案控制在 **300~600 字**，条目优先于长段落。
- `app/agents/rag_agent.py`：`max_tokens` → 300；
  提示词要求关键事实摘要 **250 字以内**，用 Markdown 要点 + `[RAG: 来源/页码]` 标注。

**效果**：综合 LLM 39s → ~8~10s；RAG 摘要 LLM 34s → ~18s（受生成速度下限约束）。

### 3.5 评审闭环提速（减少多余迭代）

- `app/agents/critique_agent.py`：收紧评审标准——仅当存在**明确硬性错误**
  （幻觉、与上下文矛盾、关键数字错误、完全未作答）才 `REVISE`，
  禁止因风格、篇幅、补充建议等主观偏好触发修订。
- `.env`：`MAX_ITERATIONS=2`，保证质量与速度平衡。

**效果**：查询迭代轮数稳定为 1 轮（多数查询评审即 PASS），
避免每多一轮叠加 20~40s 的二次综合耗时。

### 3.6 路由策略偏向 RAG（减少无关 Web 搜索）

`app/graph/workflow.py` 监督者提示词明确"内部文档知识型问题严禁路由到 web_agent"，
涉及项目/技术术语的问题默认走 RAG；仅当确需实时信息才用 web_agent / both。

### 3.7 过滤器提取默认关闭

- `app/config.py` 新增 `use_filter_extraction`（默认 `False`）。
- 收紧 `FILTER_PROMPT`：仅当用户明确提到 `source: <名称>`、`来自 <文档>`、
  `第 N 页`、`表格/图片` 才提取。
- 原因：GLM 易把普通查询误判为 source 过滤器导致 0 命中，且每次多一次 LLM 调用。

### 3.8 单例化 + 避免重复下载

- `get_qdrant_client` / `get_vector_store` / `get_sparse_embeddings` /
  `get_dense_vector_store` / `create_*_agent` 均 `@lru_cache` 单例，避免重复初始化。
- `app/rag/reranker.py`：`_model_failed` 标志——模型未缓存或加载失败后不再重复尝试下载
  （避免每次查询浪费 30s 下载）。

---

## 4. 效果对比

### 4.1 查询总耗时（连续请求，RAG 知识型问题）

| 阶段 | 优化前 | 优化后 | 说明 |
| --- | --- | --- | --- |
| 首次查询 | 96~108s | ~40~55s | 含预热兜底、首次连接 |
| 后续查询（热） | 96~108s | **30~42s** | 检索稳定在 ~5s |
| Web 类查询 | ~30s | ~30s | 路由与搜索耗时 |

### 4.2 节点耗时（热查询实测）

| 节点 | 优化前 | 优化后 |
| --- | --- | --- |
| 监督者路由 | ~1.5s | ~1.5s |
| RAG 检索 | ~20s（含失败重试） | ~5s |
| RAG 摘要 LLM | ~34s | ~18s |
| 综合 LLM | ~39s | ~8~10s |
| 评审 LLM | ~8s | ~5~8s |
| 迭代轮数 | 2~5 轮 | 1 轮 |

### 4.3 答案质量

- 综合答案长度从 1500~2500 字收敛到 **460~950 字**，更精炼、聚焦。
- 评审通过率提升，多数查询第 1 轮即 PASS。

---

## 5. 剩余瓶颈与后续可优化方向

1. **LLM 生成速度（硬下限）**：glm-4-flash 约 35~45 字符/s，
   3 个串行 LLM 调用决定了 ~30s 的下限。
   - 可选项：将 RAG Agent 的 LLM 摘要调用移除，直接把检索到的 top-k 原始片段传给综合 Agent，
     可再省 ~18s（需权衡"研究 Agent 架构"的简历亮点）。
   - 或升级到生成更快的模型 / 开启输出流式（token 级）改善感知延迟。
2. **前端 token 级流式**：当前 SSE 为节点级状态事件，最终答案一次性返回；
   若实现综合 LLM 的 `stream=True` 逐 token 推送，可显著改善"感知卡顿"。
3. **重排序器模型未缓存**：`BAAI/bge-reranker-base` 未下载，当前重排为 no-op；
   可用 `HF_ENDPOINT=https://hf-mirror.com` 预下载后启用，提升检索结果相关性。
4. **Qdrant 云地理距离**：南美节点 RTT 高（每次请求 ~4s）；
   迁移到国内/就近节点可进一步压缩检索耗时。

---

## 6. 涉及的代码文件

| 文件 | 改动 |
| --- | --- |
| `app/rag/ingestion.py` | TLS 1.2 强制 + 退避重试；单例；dense 降级 store |
| `app/rag/retriever.py` | 混合检索失败降级纯 dense |
| `app/rag/reranker.py` | 模型缓存检查 + 失败标记，避免重复下载 |
| `app/api.py` | 启动后台预热（向量库 + 重排器预检） |
| `app/graph/workflow.py` | 监督者路由偏向 RAG；节点计时日志 |
| `app/agents/rag_agent.py` | 摘要输出压缩（max_tokens=300，≤250 字） |
| `app/agents/synthesis_agent.py` | 答案输出压缩（max_tokens=700，300~600 字） |
| `app/agents/critique_agent.py` | 评审标准收紧，减少多余迭代 |
| `app/config.py` / `.env` | `USE_FILTER_EXTRACTION=false`、`MAX_ITERATIONS=2` |
