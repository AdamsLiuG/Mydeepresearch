# DeepResearch Agent

一个面向开放互联网研究任务的 Agent 系统：通过六阶段受控编排（Planning → ReAct Research → Reflection → Review → Repair → Report），结合证据质量评估、多层搜索缓存、降级策略和全链路可观测性，将复杂研究问题转化为结构化、引用可追溯的研究报告。

## 项目背景

现有的 LLM 应用大多是"一次搜索 + 直接生成"的单轮模式，面对复杂研究任务时，它们缺乏对搜索结果质量的判断、对覆盖缺口的识别、以及对最终报告的自我审查能力。

本项目构建了一个多阶段、可编排的研究 Agent，核心设计目标包括：

- **受控的多轮研究闭环**：不是搜一次就产出答案，而是通过 ReAct 证据补全、Reflection 覆盖评估、Review 质量审查和 Repair 证据修补实现有条件的多轮迭代
- **证据质量先行**：搜索结果不直接丢给 LLM，而是先经过来源分类（18 种类型）、多维评分和低质量过滤
- **工程化韧性**：搜索超时、模型输出异常、证据不足等场景都有明确的降级策略，避免 happy-path-only 的脆弱 Agent
- **全链路可观测**：每次请求都有 trace、metrics、阶段事件和成本估算，而非黑盒运行

## 核心特性

| 特性 | 说明 |
|------|------|
| **六阶段编排** | `Planning → Search/ReAct → Summarization → Reflection → Review → Report`，带条件分支和 repair 循环 |
| **任务级 ReAct 闭环** | 根据 6 维证据信号（来源数、域名覆盖、时效、质量、正文充足度）动态决定补证据动作 |
| **证据质量评估** | 18 种来源类型分类、来源可信度评分、低质量过滤（navigation page / clickbait / aggregator 检测） |
| **多层搜索缓存** | exact match + 语义缓存（sentence-transformers + ChromaDB ANN）+ 动态 TTL（fresh/normal/evergreen） |
| **降级与容错** | Planner 兜底、搜索 retry + timeout、任务级失败隔离、`partial_success` 状态、report repair loop |
| **全链路可观测** | `X-Request-ID` + `RequestTrace` + `MetricsRegistry` + SSE 实时事件 + `/metrics/json` |
| **多搜索后端** | DuckDuckGo / Tavily / SearXNG / Semantic Scholar / 多后端融合（Advanced） |
| **结构化报告** | JSON 结构化输出 → 引用校验 → Markdown 渲染，支持 fixed / flexible 两种布局模式 |
| **前端实时展示** | Vue 3 + SSE 消费，实时渲染任务卡片、来源列表、工具调用痕迹和最终报告 |
| **自建评测** | benchmark loader + heuristic judge + full system validation + perf smoke/regression/load |

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户输入研究主题                          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: Planning                                          │
│  ├─ LLM 拆解为 3~5 个结构化任务                                │
│  ├─ 4 层输出解析兜底（JSON → tool payload → md table → text）  │
│  └─ 空结果时自动生成 fallback task                              │
├──────────────────────────────────────────────────────────────┤
│  Stage 2: Task Execution (per task)                         │
│  ├─ 搜索执行（多后端 + exact/semantic cache）                   │
│  ├─ 证据入库（分类 → 评分 → 过滤 → EvidenceStore）              │
│  ├─ ReAct 闭环（Observe → Decide → Act，最多 N 轮）            │
│  │   ├─ 动作：rewrite_query / broaden_query / diversify /     │
│  │   │        fetch_page / stop                              │
│  │   └─ 信号：source_count / domain_diversity / freshness /   │
│  │            quality / body_sufficiency / gaps               │
│  └─ 任务总结（grounded findings + source_ids 绑定）             │
├──────────────────────────────────────────────────────────────┤
│  Stage 3: Reflection                                        │
│  ├─ 评估首轮覆盖度，识别 gap_signals                            │
│  └─ 不足时补充 1~2 个任务，重新执行                               │
├──────────────────────────────────────────────────────────────┤
│  Stage 4: Review                                            │
│  ├─ 规则审查：来源数、域名多样性、时效性、引用有效性                  │
│  ├─ LLM 审查（可选）：补充语义层面的证据问题                       │
│  └─ 输出 passed / warning / blocked 状态                      │
├──────────────────────────────────────────────────────────────┤
│  Stage 5: Report Repair (conditional)                       │
│  ├─ 仅针对高优先级 review issue 补 0~2 个 targeted tasks         │
│  └─ 重新 review 后输出最终报告                                  │
├──────────────────────────────────────────────────────────────┤
│  Stage 6: Report Generation                                 │
│  ├─ LLM 生成结构化 JSON 报告                                   │
│  ├─ source_id 校验 + 无效引用剔除                               │
│  └─ Markdown 渲染 + 参考来源列表                                │
└──────────────────────────────────────────────────────────────┘
```

架构图另见 [docs/assets/agent-architecture.svg](docs/assets/agent-architecture.svg)，请求生命周期见 [docs/assets/request-lifecycle.svg](docs/assets/request-lifecycle.svg)。

## 项目目录结构

```
helloagents-deepresearch/
├── backend/
│   ├── src/
│   │   ├── agent.py              # Agent 编排主文件（3451 行）
│   │   ├── main.py               # FastAPI 入口 + SSE 流式接口
│   │   ├── config.py             # 配置管理（1105 行，Pydantic BaseModel）
│   │   ├── metrics.py            # RequestTrace + MetricsRegistry
│   │   ├── prompts.py            # 所有 Agent prompt 模板
│   │   ├── models.py             # 数据模型
│   │   └── services/
│   │       ├── evidence.py       # 证据存储 + 质量评估引擎（1961 行）
│   │       ├── evidence_index.py # 证据向量索引 + 跨请求归档
│   │       ├── search.py         # 搜索调度 + 多层缓存（2341 行）
│   │       ├── planner.py        # 任务规划 + 4 层输出解析
│   │       ├── reflection.py     # 覆盖评估 + 补充研究决策
│   │       ├── reviewer.py       # 规则 + LLM 审查
│   │       ├── reporter.py       # 结构化报告生成
│   │       ├── summarizer.py     # 任务总结服务
│   │       ├── strategy_memory.py# 跨请求策略记忆（可选）
│   │       ├── note_memory.py    # 跨请求笔记记忆（可选）
│   │       └── ...
│   ├── tests/                    # 19 个测试文件
│   ├── evals/                    # benchmark + judge + runner
│   ├── perf/                     # 性能测试（smoke/regression/load/profile）
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.vue               # 前端主组件（SSE 消费 + 实时渲染）
│   │   └── services/
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml
├── .github/workflows/
│   ├── ci.yml                    # backend lint/test/build + perf + frontend build
│   └── perf-regression.yml
└── docs/
    ├── assets/                   # 架构图 SVG
    ├── DEMO_PLAYBOOK.md
    └── INTERVIEW_GUIDE.md
```

## 快速开始

### 环境要求

- Python ≥ 3.10
- Node.js ≥ 20（前端）
- 一个 OpenAI 兼容的 LLM API（本地 Ollama/LMStudio 或远程 API）

### 1. 配置后端

```bash
cd backend
cp .env.example .env
# 编辑 .env，至少配置以下字段：
# LLM_PROVIDER=custom（或 ollama / lmstudio）
# LLM_MODEL_ID=你的模型名
# LLM_API_KEY=你的 API Key
# LLM_BASE_URL=你的 API 地址
# SEARCH_API=duckduckgo（默认，无需额外 API Key）
```

### 2. 安装并启动后端

```bash
# 使用 uv（推荐）
uv sync
uv run python src/main.py

# 或使用 pip
pip install -e ".[dev]"
python src/main.py
```

后端默认监听 `http://localhost:8000`。

### 3. 安装并启动前端

```bash
cd frontend
npm ci
npm run dev
```

前端默认开发端口 `5174`，自动代理请求到后端 `http://localhost:8000`。

### 4. 使用 Docker（可选）

```bash
docker-compose up --build
```

### 5. 发起研究请求

在前端界面输入研究主题，或直接调用 API：

```bash
# 同步接口
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "大语言模型推理优化的关键技术与最新进展"}'

# 流式接口（SSE）
curl -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"topic": "大语言模型推理优化的关键技术与最新进展"}'
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 健康检查 |
| `GET` | `/metrics/json` | 全局 metrics 快照（counters, latencies, cost, recent requests） |
| `POST` | `/research` | 同步研究请求 |
| `POST` | `/research/stream` | SSE 流式研究请求 |
| `GET` | `/requests` | 已持久化的请求列表 |
| `GET` | `/requests/{id}` | 获取单个请求快照 |
| `POST` | `/requests/{id}/resume` | 恢复未完成的研究请求 |
| `POST` | `/requests/{id}/resume/stream` | SSE 流式恢复 |

## 核心实现说明

### Agent 主流程编排

`DeepResearchAgent`（[agent.py](backend/src/agent.py)）是系统核心编排器，管理 6 个专职 Agent（规划、总结、反思、审查、ReAct 决策、报告）的协作。

编排不是简单的线性 pipeline：
- **Planning** 完成后，如果输出为空或格式无法解析，会自动生成 fallback task 继续执行
- **ReAct** 闭环在每个任务内运行，根据当前证据质量信号决定是否继续补证据
- **Reflection** 在所有任务完成后评估覆盖度，不足时规划补充任务并重新执行
- **Review** 结合规则审查和可选的 LLM 审查，产出结构化问题列表
- **Repair** 仅在 review 发现高优先级问题时触发，运行 0~2 个定向修补任务

### 任务级 ReAct 证据补全

每个任务在首轮搜索后，进入一个受控的 ReAct 小循环（[agent.py L122-154](backend/src/agent.py)）：

1. **Observe**：`TaskReactObservation` 计算 6 维证据信号
   - `source_count`：当前来源总数
   - `source_diversity`：不同域名数
   - `freshness_ok`：是否有近期来源
   - `evidence_sufficiency`：正文是否充足
   - `gap_signals`：缺口列表
2. **Decide**：先尝试 `_fallback_task_react_decision()`（规则兜底），再用 LLM `_plan_task_react_decision()` 选择动作
3. **Act**：执行 `rewrite_query` / `broaden_query` / `diversify_source_query` / `fetch_page` 中的一个
4. **终止条件**：达到最大轮次、预算耗尽、或 LLM 判断 `stop`

### 证据质量评估引擎

[evidence.py](backend/src/services/evidence.py)（1961 行）实现了一个完整的来源评估管线：

- **来源类型分类**（18 种）：government / education / peer_reviewed_paper / preprint_paper / official_documentation / repository / news_primary / news_secondary / forum_expert / social_official / content_farm_or_aggregator 等
- **多维评分**：基础分 + 来源类型加分（`SOURCE_TYPE_BONUS`，如 government +4.0、social_general -1.0）+ provider 交叉验证加分 + 时效标签
- **质量过滤器**：
  - `is_navigation_or_listing_page()`：检测纯导航/列表页
  - `is_clickbait_or_low_information_page()`：clickbait 标题模式匹配
  - `is_aggregator_or_rehosted_copy()`：聚合站/转载检测
  - `is_duplicate_or_near_duplicate()`：内容重复检测
  - `estimate_extraction_quality()`：正文提取质量分级（good / partial / poor）

### 搜索与缓存机制

[search.py](backend/src/services/search.py)（2341 行）实现了三层缓存：

1. **Exact Match**：标准化 query + search_api + fetch_full_page 后的精确匹配，支持 diskcache 持久化
2. **Semantic Cache**：sentence-transformers 编码 + ChromaDB ANN 向量检索，按 cosine similarity 阈值判定命中
3. **Lexical Fallback**：当 embedding 不可用时，退化到 Jaccard n-gram 相似度匹配
4. **Dynamic TTL**：根据 query 的时效性信号（"最新" / "2025" / "overview" / "protocol"）自动分配 fresh/normal/evergreen 三档 TTL

### 报告生成与引用校验

[reporter.py](backend/src/services/reporter.py)（1148 行）生成结构化报告：

- LLM 输出 JSON 格式（背景概览 / 核心洞见 / 证据与数据 / 风险与挑战 / 自定义章节）
- 每条结论必须绑定 `source_ids`，系统自动校验有效性，无效引用被剔除
- 支持 `flexible`（核心框架 + 动态自定义章节）和 `fixed`（经典固定结构）两种布局模式
- 内部流程语言过滤：自动剔除 "blocked" / "审查提示" / "source_id 校验" 等系统话术，确保报告面向读者

### LLM 兼容层

`SafeHelloAgentsLLM` 解决本地模型和不同 provider 的输出不稳定问题：

- **响应回退**：`content → reasoning → reasoning_content`，三层尝试提取有效文本
- **流式聚合**：streaming 模式下先收集 visible content，无 content 时 fallback 到 reasoning buffer
- **代理绕过**：自动检测本地/私有网络 LLM endpoint，绕过 HTTP 代理环境变量
- **非字符串响应处理**：`_coerce_text()` 递归处理 list / dict / model_dump 等异构响应格式

## 技术栈

| 层 | 技术 |
|---|------|
| Agent 框架 | [HelloAgents](https://github.com/hello-hq/hello-agents)（`ToolAwareSimpleAgent`） |
| 后端 | FastAPI + Uvicorn |
| LLM 接入 | OpenAI-compatible API（支持 Ollama / LMStudio / vLLM / 任意兼容服务） |
| 搜索 | DuckDuckGo / Tavily / SearXNG / Semantic Scholar / Multi-backend Fusion |
| 向量检索 | sentence-transformers + ChromaDB |
| 缓存 | diskcache（持久化）+ 内存缓存 |
| 前端 | Vue 3 + TypeScript + Vite |
| 容器化 | Docker + docker-compose |
| CI | GitHub Actions（lint + test + build + perf smoke + load） |
| Linting | Ruff |
| 测试 | pytest（19 个测试文件） |

## 项目亮点

> 以下亮点均可在源码中找到对应实现。

### 1. 多层 Agent 编排，而非线性 Pipeline

系统不是"搜索 → 总结 → 出报告"的单次流程，而是包含 `ReAct 证据补全 + Reflection 覆盖评估 + Review 质量审查 + Repair 证据修补` 的多层闭环。每个闭环都有明确的进入/退出条件和预算控制。

### 2. 证据质量评估驱动的 ReAct 决策

ReAct 闭环不是盲目重试，而是基于 6 维证据信号做受控决策：来源数不足时 broaden_query，域名单一时 diversify_source_query，正文缺失时 fetch_page。这比简单的"搜索失败就重试"更精细。

### 3. 18 种来源分类 + 低质量过滤器

不直接把搜索结果丢给 LLM。证据入库前先经过来源类型分类、可信度评分和多种过滤器（导航页检测、clickbait 检测、聚合站检测、重复检测），减少低质量信息对最终报告的干扰。

### 4. 三层搜索缓存 + 动态 TTL

exact match → 语义 ANN → lexical fallback 三层匹配，避免相似 query 的重复搜索。TTL 根据 query 中的时效信号自动分桶（搜"2025最新进展"→ 短 TTL，搜"TCP 协议原理"→ 长 TTL）。

### 5. 工程化降级策略

- Planner 输出不可解析 → 尝试从 tool_call / markdown table / 编号文本中恢复任务
- Planner 完全无输出 → 自动生成 fallback task
- 搜索超时 → 可配置 retry + backoff
- 单任务失败 → 标记 `failed` 但请求继续，最终标记 `partial_success`
- 报告审查发现严重问题 → 触发 repair 循环补证据

### 6. 全链路可观测性

每个请求自带 `X-Request-ID`，`RequestTrace` 记录每个阶段的延迟、状态、token 消耗和成本估算。`/metrics/json` 暴露全局 counters（包括 cache hit rate、ReAct stop reasons、reflection/review 统计）。前端 SSE 实时推送 `stage_started/completed`、`task_status`、`metrics_snapshot` 等事件。

### 7. LLM 输出鲁棒性处理

`SafeHelloAgentsLLM` 处理本地模型的响应不稳定问题（空 content、reasoning-only 输出、非标准 JSON 格式）；Planner 输出解析有 4 层 fallback（JSON → tool call payload → markdown table → numbered text），最大程度从不规范的 LLM 输出中恢复可用结果。

### 8. 自建评测与 CI

benchmark 定义、批量运行、heuristic judge 评分全部自建，不依赖第三方评测框架。CI 覆盖后端 lint / test / build / perf smoke / perf load / 前端 build，保证核心路径不因改动退化。

## 可选能力

以下功能代码已完整实现，默认配置为关闭，可按需启用：

- **Strategy Memory**（`STRATEGY_MEMORY_ENABLED`）：从历史请求中提炼检索策略与失败模式，跨请求复用
- **Note Memory**（`NOTE_MEMORY_ENABLED`）：基于向量检索的历史笔记记忆，辅助 planning 和 task execution
- **Evidence Memory**（`EVIDENCE_MEMORY_ENABLED`）：证据归档与跨请求证据复用
- **Advanced Search Fusion**（`SEARCH_API=advanced`）：多搜索后端并行 + 结果融合 + 可选 LLM reranking

## 配置说明

所有配置通过环境变量管理，完整配置模板见 [backend/.env.example](backend/.env.example)。关键配置分类：

| 分类 | 核心变量 |
|------|---------|
| **LLM** | `LLM_PROVIDER`, `LLM_MODEL_ID`, `LLM_API_KEY`, `LLM_BASE_URL` |
| **搜索** | `SEARCH_API`, `FETCH_FULL_PAGE`, `SEMANTIC_SCHOLAR_API_KEY` |
| **Agent 编排** | `MAX_AGENT_TASKS`, `TASK_REACT_ENABLED`, `TASK_REACT_MAX_ROUNDS`, `REQUEST_REFLECTION_ENABLED`, `REVIEW_STAGE_ENABLED`, `REPORT_REPAIR_ENABLED` |
| **搜索缓存** | `SEARCH_CACHE_ENABLED`, `SEMANTIC_CACHE_ENABLED`, `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` |
| **工具守卫** | `SEARCH_TOOL_TIMEOUT_SECONDS`, `SEARCH_TOOL_RETRY_ATTEMPTS` |

## Benchmark / Perf / CI

### Offline Benchmark

```bash
cd backend
python evals/run_benchmark.py \
  --input evals/benchmarks/sample_benchmark.jsonl \
  --output evals/results/sample_results.json
```

### Performance Testing

```bash
cd backend
# Smoke test（stub 模式，验证链路）
python -m perf.run_smoke --profile stub

# Regression test（real_local 模式，写入 baseline）
python -m perf.run_regression --profile real_local --write-baseline

# Load test
python -m perf.run_load --profile stub --users 4 --spawn-rate 2 --duration 20s

# Profile test
python -m perf.run_profile --profile real_local
```

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) 覆盖：
- Backend: Ruff lint → pytest → build check
- Perf: smoke benchmark → load smoke
- Frontend: npm ci → build check

## 局限性与后续优化

### 当前局限

- **依赖外部搜索质量**：最终报告质量受搜索后端返回结果的影响，DuckDuckGo 免费接口在某些主题上覆盖有限
- **长链路延迟较高**：完整六阶段 + ReAct 的端到端延迟在数分钟级别，主要瓶颈在串行搜索和多次 LLM 调用
- **评测体系待完善**：当前只有 heuristic judge（keyword coverage + section completeness），缺少 LLM-as-Judge 对事实准确性的评估
- **agent.py 文件过大**：编排主文件 3451 行，虽然内部按职责分区，但文件级别的模块化程度有提升空间
- **缺乏并发优化**：任务执行目前为串行，可通过任务级并行提升吞吐
- **Memory 模块未充分验证**：Strategy Memory / Note Memory 代码完整但默认关闭，跨请求学习效果缺少系统评估

### 后续优化方向

- [ ] 增加 LLM-as-Judge，评估报告的事实准确性和引用质量
- [ ] 跑 baseline（关闭 ReAct/Review/Repair）vs full pipeline 的对比实验，量化各模块贡献
- [ ] 拆分 agent.py 为 task_executor / search_orchestrator / stream_coordinator
- [ ] 增加任务级并行执行
- [ ] 增加 demo GIF / 截图
