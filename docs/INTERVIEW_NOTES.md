# OmniRAG — 面试难点复盘与话术素材库

> 使用说明：按模块组织，每节对应一个高频面试场景题。
> 每条都以「场景提问 → 根因 → 解决 → 改进空间」的结构呈现，避免只讲「修了什么」不讲「为什么」。
> 代码引用均指向实际文件 + 行号，便于被追问时直接翻源码佐证。

---

## 一、多智能体编排（LangGraph）

### 场景题 1：“你说这是多智能体，不就是 A 调完调 B 的顺序流程吗？”

**核心回答**：本项目的执行图包含 **动态条件路由 / 并行分支 / 循环边** 三种非顺序结构，而非线性流水线。

1. **动态条件路由（控制流交给 LLM，代码做保险丝）**
   - 监督者节点（`app/graph/workflow.py:79-120`）把「对话历史 + 查询 + 已有的 RAG/Web 上下文」喂给 LLM，自由文本选择 `rag_agent / web_agent / both / synthesis`，经 `add_conditional_edges` 分发。
   - 两层防御：① Prompt 层明确约束（上下文为空时绝不能选 `synthesis`）；② 代码层 `route_supervisor`（`app/graph/workflow.py:214-229`）再做兜底校验，上下文真的为空就强制改道 `both`。
   - **改进空间**：路由输出改用 `with_structured_output` 或 function calling 强约束；当前自由文本 + 兜底是明确的取舍——智谱 `with_structured_output` 偶发返回 None，在路由这种单点路径上不够可靠。

2. **真并行：单节点 asyncio.gather 与 LangGraph 原生 fan-out 的取舍**
   - `both_node`（`app/graph/workflow.py:137-147`）在一个节点内用 `asyncio.gather` 并发执行 RAG 检索与 Web 搜索，而不是从 supervisor 拉出两条边到两个节点再汇聚。
   - 取舍依据：两条分支写的是**不同状态字段**（rag_context / web_context），单节点内 gather 的状态语义更干净、SSE 事件更好发（一个 `both` 事件）；代价是图拓扑看不出并行意图。
   - 关键坑：底层所有 LLM 调用是同步的，必须包 `asyncio.to_thread`（`app/graph/workflow.py:107` 等处），否则“并行”会退化成串行。

3. **评审闭环：带反馈的循环边且反馈真正起效**
   - `synthesis_node → critique_node →（REVISE 则回到）synthesis_node` 构成图里的循环边（`app/graph/workflow.py:270-274`）。
   - 综合 Agent prompt（`app/agents/synthesis_agent.py:31-32`）明确要求：如有评审反馈必须逐条解决；若反馈为 GOOD 则保持质量不做无意义改写。
   - `iterations` 状态计数器保证有界，避免死循环烧 token。
   - 解析坑：`_critique_passed()`（`app/graph/workflow.py:194-209`）需先排除 `"NOT GOOD"` 等反例子串，再取最后一行判定；评审输出还存在多种格式（直接 `GOOD` 或“评估：…评分：GOOD”）。
   - **改进空间**：评审输出结构化（`{"passed": bool, "issues": []}`），让类型系统取代字符串考古。

4. **记忆与流式**
   - SQLite `AsyncSqliteSaver` + `thread_id` 持久化对话历史，截窗后进入监督者与综合 prompt，支持追问沿用上一轮信息源。
   - 坑：checkpoint 数据目录必须存在，否则工作流初始化会失败。
   - `graph.astream(stream_mode="updates")` 将节点增量输出转成 SSE 事件（route/rag/web/both/synthesis/critique/final），前端可实时渲染 Agent 执行过程。

**追问答法**：当前架构是 **supervisor 星型**（中心路由、叶子执行，agent 间不通信）。更高级的网状协作是 agent-as-tool：RAG Agent 检索后发现覆盖不足，**主动**用 ReAct 把问题委托给 Web Agent。LangGraph 可用 Command 模式让节点动态决定下一节点。v2 方向明确，但 v1 的价值在于先用最可控的结构把闭环和并行跑对。

---

## 二、RAG 内核

### 场景题 2：“混合检索不就是向量库一个 API 的事吗？”

**核心回答**：本项目是 **双路召回（Dense/Sparse RRF）+ 查询理解层（过滤器/多查询）+ 交叉编码器重排** 的完整管线，关键细节和坑在工程层：

1. **双路召回与服务端 RRF**
   - **Dense**：智谱 embedding-3，1024 维 COSINE；
   - **Sparse**：FastEmbed `Qdrant/bm25`，由于返回 `SparseEmbedding` 对象与 LangChain 接口不兼容，写了 `FastEmbedSparseAdapter` 适配器（`app/rag/ingestion.py:60-73`）；
   - 融合用 Qdrant 原生 `RetrievalMode.HYBRID`（`app/rag/ingestion.py:215-226`），RRF 在服务端算，避免拉全量候选。
   - ~~**KG 第三路**~~（已移除，2026-09）：曾用 Neo4j 一跳邻居拼接伪文档，成本高收益低已删除；双路合并里的 **MD5 去重**（而非 `hash()`）做法沿用——Python 内置 hash 受 `PYTHONHASHSEED` 影响每次进程重启都不同。

2. **查询理解层：结构化过滤器 + 多查询扩展**
   - LLM 提取 `RAGFilters(source/content_type/page)`（`app/rag/retriever.py:30-52`），prompt 明确「务必保守，仅当用户明确提及时才设置」——假阳性比假阴性伤害大。
   - **遗留坑**：提取出的过滤词（如 “report.pdf 第3页”）**没有从查询文本剥离**，会带着噪音做相似度检索，轻微拉偏召回；改写查询再检索是正解。
   - Multi-Query 扩展时构造 `use_multi_query=False` 的实例防止**无限递归**（`app/rag/retriever.py:82-92`）。

3. **交叉编码器重排的触发与静默降级**
   - 触发条件：`use_reranking and len(candidates) > reranker_top_n`（`app/rag/retriever.py:127`）——候选少时重排浪费推理时间。
   - 坑：bge-reranker-base 权重下载失败（国内网络镜像问题）时**静默降级为截断**，保证检索主链路可用。**改进空间**：降级要打明确 warning 日志或暴露可观测指标，否则退化不可见。

4. **KG 检索（已移除，2026-09）**
   - 原实体匹配为关键词 `CONTAINS` + jieba 分词（`app/rag/kg.py`），本质是「关键词 + 一跳邻居」，并非真正图推理，成本高收益低，已整块删除（`app/rag/kg.py`、`neo4j`/`jieba` 依赖、`NEO4J_*` 配置、`USE_KG_RETRIEVAL` 开关全部移除）。
   - **技术演进方向**：若未来重上图谱，实体链接应走 embedding 相似度（query 向量 vs 实体 description 向量），关键词包含在同义词场景（如“大模型”匹配不到“GLM-4”）必丢召回——图谱节点本身应建向量索引。

---

## 三、LangSmith 可观测性

### 场景题 3：“Agent 链路这么复杂，出了问题怎么排查？”

**核心回答**：LangChain 生态的自动埋点——只需要两个环境变量 `LANGCHAIN_TRACING_V2=true`、`LANGCHAIN_API_KEY`（`app/config.py:24-25`）。业务代码零改动，以下内容全部自动上报 LangSmith：

- LangGraph 每个节点进出的完整状态快照；
- 每次 LLM 调用的完整 prompt / response / token 用量；
- 每个 Retriever 的召回结果列表；
- SQLite checkpoint 前后的对话历史。

典型排查场景：

| 现象 | LangSmith 排查路径 |
|---|---|
| 路由错误（不该走 Web 却走了） | 直接看监督者当时的完整输入（历史 + 上下文快照），判断是 prompt 缺陷还是 LLM 抽风 |
| 答案质量差 | 逐层展开：检索召回为空？重排挤掉了正确答案？还是综合阶段幻觉？ |
| 评审 REVISE 迭代无改善 | 对照两轮 synthesis 输入的差异，看 critique 是否被正确传入并响应 |

**改进空间**：
- 显式设 `LANGCHAIN_PROJECT` 区分 dev/prod，避免 trace 全混在默认项目；
- 给关键节点打 metadata（如 `{"route": "both", "iteration": 2}`），才能做跨会话聚合分析；
- 用 LangSmith Datasets & Experiments 沉淀线上 bad case 为回归测试集，prompt 改完跑一遍对比指标。

---

## 四、评估体系

### 场景题 4：“你怎么证明你的 RAG 系统是好的？”

**核心回答**：当前采用**轻量评估**——人工黄金集 `data/golden_set.json` + 检索 `recall@k` + LLM-as-judge（`faithfulness` / `answer_relevancy`），命令 `python -m app.rag.evaluation`，报告存 `data/eval_reports/eval_<时间戳>.json`。早期曾用 RAGAS 四指标端到端评估，因依赖重（ragas / datasets / pyarrow）、pyarrow pickle 崩溃、judge 与生成同模型自引用失真等问题，已在 2026-09 重构中移除。评估方法论踩过的坑（保留为技术演进叙述）：

1. **评估上下文必须是原始 chunk**
   - 早期 RAGAS 的 Context Precision / Recall 衡量的是**检索**质量，如果用 RAG Agent 加工后的摘要去评，测的是综合 Agent 的压缩能力而不是检索质量——**指标会系统性虚高**。因此专门写了 `_retrieve_raw_contexts()` 走裸检索器。现轻量评估的 `recall@k` 同样只针对检索层：取回 top-k 原始 chunk，用智谱 embedding 算 ground_truth 与各 chunk 的最大余弦相似度、超过阈值记命中。

2. **评测集从「LLM 自动生成」改为「人工黄金集」**
   - 早期从向量库采样文档让 LLM 生成 `(query, ground_truth)` 对，prompt 要求事实型/比较型/推理型多样化。坑：智谱 `with_structured_output` 偶发返回 None，改用普通 JSON 输出 + 正则解析。
   - 现改为 `data/golden_set.json` **人工校验**样本，避免系统「自产自销」的自引用失真。

3. **工程坑：pyarrow pickle 崩溃（RAGAS 时代，已随重构消除）**
   - 报错：`Can't pickle <class 'MonthDayNano'>`——RAGAS 默认多线程，`relativedelta` 类型跨进程序列化失败。
   - 解：`RunConfig(max_workers=1)` 单线程；同时评估时把 Self-Critique 循环降为 `max_iterations=1`——**评估测的是“检索 + 一次生成”的质量**，不该把修订循环的成本也算进去，指标语义要对齐但不与生产完全相同。
   - 现轻量评估不再依赖 ragas / datasets / pyarrow，该问题自然消失。

---

## 五、知识图谱与可视化（KG 已移除，仅作技术演进参考）

> ⚠️ **现状**：Neo4j 知识图谱已在 2026-09 重构中**整体移除**——原 kg_retrieve 只是「jieba 关键词 + 一跳邻居」，并非真正图推理，成本高收益低；`app/rag/kg.py`、`neo4j`/`jieba` 依赖、`NEO4J_*` 配置、`/api/graph` 与图谱可视化页均已删除。以下踩坑经历可作为面试里「技术演进 / 做减法」的叙述保留，但结论需先说清「KG 已移除」。

### 场景题 5：“Neo4j 连不上你怎么处理的？”

**核心回答**：**协议级迁移**而非在 Bolt 上打补丁。

- 现象：Neo4j Aura Bolt 协议（7687 端口）反复报 `Unable to retrieve routing information`，换 `neo4j+ssc` 跳证书、加超时、改 Clash DIRECT 规则都无效。
- 根因：国内网络对 7687 非常用端口 TLS 握手被劫持/RST，Aura 443 只跑 HTTPS 不跑 Bolt。
- 解：改用 Aura 官方 **HTTP Query API v2**（`/db/{db}/query/v2`，443 端口 HTTPS），自封装 `aura_query()`（`app/rag/kg.py:88-115`）：Basic Auth + 3 次重试 + records 解析；写入改为 UNWIND 批量 MERGE（`app/rag/kg.py:141-173`）；写入/检索/可视化三个消费方全量迁移。
- 副作用：代理链路对 443 的 MITM 使证书校验失败，`verify=False` 为明确妥协；**生产建议在可直连网络恢复证书校验**（已写进 README 已知限制）。
- 调试教训：应先用 `openssl s_client` 等做**网络分层诊断**，怀疑代码之前先验证链路。
- **结论**：该模块现已在 2026-09 整体移除；此经历的价值在于「协议级迁移」的解决思路与「网络分层诊断」的方法论，可作为技术演进叙述。

### 场景题 6：“图谱能生成但用户看不到——为什么静态快照不够？”

- 现象：`scripts/visualize_kg.py` 生成的 `knowledge_graph.html` 是**数据内嵌的一次性快照**，能离线打开但无法实时更新，与“上传文档后自动看到新图谱”的预期冲突。
- 解：React 新增 GraphPage（vis-network 交互式可视化），后端新增 `/api/graph` 实时拉取节点关系；上传导入完成后前端自动跳转图谱页。数据截断（LIMIT）防大图浏览器卡死。
- 改进空间：静态快照与实时 API 应该并存（快照用于分享），但快照要打**时间戳水印**——否则极易产生“图谱里是旧数据”的认知错位。
- **结论**：图谱可视化（GraphPage、`/api/graph`、`scripts/visualize_kg.py`）已随 KG 在 2026-09 一并移除；当前前端仅保留 Chat/Data/Eval 三页。

---

## 六、基础设施与网络层（GFW 时代的云服务矩阵）

### 场景题 7：“外部依赖全挂了项目还能跑吗？”

**核心回答**：三个云服务各有各的毛病，对应三种不同层次的处理策略——**承认网络是架构的一部分**。

| 服务 | 症状 | 处理手段 | 层次 |
|---|---|---|---|
| Neo4j Aura（已移除，2026-09） | Bolt 7687 被劫持 | 原换官方 HTTP Query API（443）；现模块已整体移除 | ~~协议级迁移~~ |
| Qdrant Cloud（南美区） | SSL EOF，直连干扰 | monkey-patch 客户端内部 `ApiClient.send_inner`，强制用配了代理的 httpx.Client（`app/rag/ingestion.py:78-88`） | **库内部件级 patching** |
| HuggingFace bge-reranker | 模型下载 403/超时 | 镜像源 `HF_ENDPOINT=hf-mirror.com` 或本地权重路径 | **资源源级切换** |

讲这个问题的面试加分项：**对 monkey-patch 的反思**——它是最后手段，版本升级即失效，因此 patch 集中在一处并加注释说明版本依赖。**可控的技术债比优雅的空想更有价值**。

### 场景题 8：“后端 200 响应里缺字段，前端整页白屏”

**场景还原**：知识库页面白屏，控制台 `Cannot convert undefined or null to object`；但直接 `fetch('/api/stats')` 数据正常——陷入“代码没问题但就是崩”的矛盾。

**根因（双 bug 叠加）**：
1. Qdrant Cloud SSL 偶发中断，后端 except 分支返回 `200 + {"error": "..."}`——**状态码 200 但缺 sources 字段**；
2. 前端 `Object.keys(stats.sources)` 没有 `?.` 防御。
3. 另一独立 bug：上传组件内部用了未声明的 `onNavigate` prop，渲染即 ReferenceError。

**调试要点**：
- 浏览器报错行号是**编译后代码行号**，不等于源码行号——别被 sourcemap 偏差迷惑；
- 这种“偶发”问题优先看**后端 warning 日志**而非前端网络面板，直测接口时后端恰好没抖是最大陷阱。

**解**：前后端双层防御——后端错误分支也返回**完整契约结构**（空 sources + error 字段）；前端所有跨字段访问改可选链 `?.`。组件 prop 补声明补传递。

**改进空间**：① 错误响应用 HTTP 状态码 + FastAPI 统一异常模型，不用“200 + error 字段”；② 前端对 API 响应做 zod 运行时校验，在开发期就拦截契约漂移。

---

## 七、文档解析（重型依赖装不上怎么办？）

### 场景题 9：“docling 装不上，PDF / DOCX 还能导入吗？”

**核心回答**：docling 作为**可选增强**，降级路径为：
- PDF：PyMuPDF 按页提取文本+表格；
- DOCX：zipfile 标准库解 docx 包，解析内部 XML；
- TXT/MD：直接读取。

零硬依赖，安装了 docling 自动升级解析质量。

**改进空间**：对可选依赖的探测集中为 capability 探测函数，放在启动时而非请求时，避免每个调用点散落 try/import。

---

## 八、LLM 结构化输出

### 场景题 10：“结构化抽取（实体/过滤器/评测集）会丢数据吗？”

**核心回答**：智谱 GLM `with_structured_output` 偶发返回 `None`（不抛异常、不给结构），是非常隐蔽的 silent failure。

**处理策略分级**：
- 轻度路径（~~KG 实体抽取~~，KG 已移除）：原返回 None 时重试一次，兜底空结构（`app/rag/kg.py` 已删除）；
- 中度路径（过滤器提取）：保守提取，失败结果全为 null 反而更安全；
- 重度路径（评测集生成）：早期用 RAGAS 时代曾要求 JSON 文本 + 正则解析；现轻量评估改用**人工黄金集**，不再由系统生成评测集。

**改进空间**：生产级做法是 Pydantic 校验失败的 chunk 进**补跑队列**，把“抽不出来”变成可观测可补的闭环，而不是尽力而为。

---

## 九、三句话总结项目在简历上的差异化

1. **工程化 RAG，不是 demo RAG**：双路召回 + Cross-Encoder + 元数据过滤器，每一步都有工程细节和坑（去重策略、重排触发、过滤器保守性），而不是三行 LangChain 代码。
2. **真正的多智能体闭环**：LangGraph 条件边 + 循环边 + 并行分支，评审反馈真正传回去且被响应，不是伪闭环。
3. **基础设施级问题处理与技术做减法**：Qdrant monkey-patch、文档解析依赖降级，以及果断移除成本高收益低的 KG（Neo4j）与 RAGAS 重依赖、改为人工黄金集 + recall@k + LLM-as-judge 轻量评估——每一个都能对应一个面试官场景题，体现的是“在国内真实环境跑通云服务全链路、并敢于对不划算的组件做减法”这种真实经验，而不是教程里一帆风顺的 happy path。
