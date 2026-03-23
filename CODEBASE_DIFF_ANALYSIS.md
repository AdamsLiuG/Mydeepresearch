# `helloagents-deepresearch` 代码库差异分析

## 1. 分析范围

本文对比的两个项目分别是：

- 升级版项目：`/media/main/hjz/agent/deepresearch/helloagents-deepresearch`
- 原始项目：`/media/main/hjz/agent/hello-agents/code/chapter14/helloagents-deepresearch`

本次对比聚焦于会真实影响行为或工程交付质量的内容：

- `backend/src`
- `frontend/src`
- 依赖与打包配置文件
- 测试、评测、文档、Docker、CI

以下生成物或运行产物不作为核心实现差异来分析：

- `frontend/node_modules`
- `frontend/dist`
- `backend/build`
- `backend/notes`
- `__pycache__`

## 2. 执行摘要

升级版项目并不是一次重写。它保留了原有 API 形态以及前后端协作方式，然后在教程级 Demo 的基础上补了一层工程化加固。

最重要的结论可以概括为：

> 升级版的价值并不主要来自“功能更多”，而是来自“同一套研究流程终于更像一个可追踪、可测试、可复现的工程项目”。

从高层看，升级版主要做了这些事情：

- 保留 `/research` 和 `/research/stream`
- 增强请求级追踪与可观测性
- 增加部分失败止损与降级行为
- 通过进程内缓存减少重复搜索
- 改善配置加载与运行时默认值
- 把后端指标可视化到前端
- 增加回归测试、离线评测、Docker、CI 和项目级文档

另外还有两个很重要的观察：

- 从文件层面看，升级版基本是原始项目的超集
- 原有源码文件没有被删除
- 升级版目录中提交了更多生成物，这有利于演示复现，但也会让仓库更重、diff 更噪

## 3. 主要结构性新增

相较于原始版本，升级版新增了这些有明确工程价值的文件或目录：

- `CODEBASE_DIFF_ANALYSIS.md`（本文档）
- `README.md`
- `ENGINEERING.md`
- `docker-compose.yml`
- `backend/Dockerfile`
- `backend/src/metrics.py`
- `backend/tests/test_config.py`
- `backend/tests/test_agent.py`
- `backend/tests/test_api.py`
- `backend/tests/test_metrics.py`
- `backend/tests/test_search_cache.py`
- `backend/tests/test_evals.py`
- `backend/evals/loader.py`
- `backend/evals/schema.py`
- `backend/evals/runner.py`
- `backend/evals/run_benchmark.py`
- `backend/evals/judges/base.py`
- `backend/evals/judges/heuristic.py`
- `.github/workflows/ci.yml`

这些新增项非常清楚地表明，项目已经从“演示代码”转向“可交付项目”。

## 4. 接口与协作层变化

原有请求流程被保留下来，这与“最小侵入式升级”的目标是一致的。

保持兼容的部分有：

- 原有 `/research`
- 原有 `/research/stream`
- 原有前端流式交互流程
- 原有四阶段研究结构：`planning -> search -> summarization -> report`

在不破坏原契约的前提下，新增了这些能力：

- 在 `backend/src/main.py` 中新增 `GET /metrics/json`
- 所有 HTTP 响应增加请求级 `X-Request-ID`
- SSE 事件类型更加丰富：
  - `stage_started`
  - `stage_completed`
  - `fallback_triggered`
  - `degraded_response`
  - `metrics_snapshot`

这是一种很重要的设计取舍：升级版强化了可观测性和可排障性，但没有推翻原有交互模型。

## 5. 后端源码变化

### 5.1 执行正确性修复

`backend/src/agent.py` 中有一个非常关键的修复，它不只是重构，而是实实在在修掉了同步路径上的执行问题。

在原始版本的同步 `run()` 路径里，代码调用了 `_execute_task(...)`，但没有消费这个生成器。这意味着同步路径下任务体实际上不会真正执行。

原始写法：

- `self._execute_task(state, task, emit_stream=False)`

升级后写法：

- `for _ in self._execute_task(state, task, emit_stream=False): pass`

这个改动不是表面优化，而是真正修复了执行 bug，并且新增了 `backend/tests/test_agent.py` 来回归验证。

### 5.2 更安全的 LLM 处理

升级版在 `backend/src/agent.py` 中新增了 `SafeHelloAgentsLLM`。

它的目标是让本地模型或 OpenAI-compatible 模型在响应格式不稳定时更稳一些，具体包括：

- 从 `content` 中提取文本
- 没有 `content` 时回退到 `reasoning`
- 再回退到 `reasoning_content`
- 统一处理空响应或非字符串响应
- 使用流式聚合来避免脆弱的同步读取假设

这是一个很务实的兼容层，尤其适合本地 vLLM 或响应结构不完全统一的模型服务。

### 5.3 请求级 Trace 与 Metrics

新增的 `backend/src/metrics.py` 是这次工程化升级里最重要的模块之一。

它引入了两个核心概念：

- `MetricsRegistry`：进程内聚合计数器与延迟统计
- `RequestTrace`：单次请求级 trace，记录阶段事件、任务结果、缓存命中、token 使用和成本估算

新增的聚合指标包括：

- 请求总量与 success/partial_success/failed 计数
- fallback 总次数
- LLM 调用总量与失败次数
- 搜索调用总量与失败次数
- cache hit/miss 总量
- prompt/completion/total token 数
- estimated cost
- 各阶段 latency 统计

这些能力同时服务于：

- 后端 `/metrics/json`
- 前端通过 `metrics_snapshot` 展示的实时指标卡片

### 5.4 失败隔离与降级行为

原始版本的链路更接近“中间任何一步失败，都可能把整条请求带崩”。

升级版改成了任务级失败隔离：

- planner 没有产出任务时，自动回退到 fallback task
- 搜索失败时，把任务标记为 `failed`，而不是直接让整个请求崩溃
- 总结失败时，同样把任务标记为 `failed`
- `skipped` 和 `failed` 任务会反映到 `partial_success`
- degraded 状态会明确发到 SSE 事件流里

这对长链路 agent workflow 很重要，因为它意味着单个依赖不稳定时，不再等于整个请求必然失败。

### 5.5 搜索缓存与 Prompt 预算控制

`backend/src/services/search.py` 的改动幅度很大。

核心新增点包括：

- 进程内搜索缓存
- TTL 支持
- cache hit/miss 统计
- 与 observer/metrics 的集成
- 可配置的 `max_tokens_per_source`
- 直接答案裁剪
- 任务上下文裁剪

这些改动的优化意图非常明确：

- 避免相同查询重复搜索
- 避免 prompt 上下文膨胀
- 让上下文大小更接近模型预算

这是一种典型的工程优化，而不是单纯加功能。

### 5.6 配置层增强

`backend/src/config.py` 已经从一个相对简单的运行时配置对象，演进成更像正式项目 settings 层的实现。

新增行为包括：

- 自动加载 `.env`
- `search_api` 规范化
- `log_level` 规范化
- `cors_origins` 规范化
- pricing JSON 字符串解析与规范化
- 将相对 `NOTES_WORKSPACE` 解析到后端根目录下
- 新增 cache、metrics、上下文预算、并发、pricing 等配置项

代码实际暴露出的新增配置示例包括：

- `HOST`
- `PORT`
- `LOG_LEVEL`
- `CORS_ORIGINS`
- `SEARCH_CACHE_ENABLED`
- `SEARCH_CACHE_TTL_SECONDS`
- `METRICS_RECENT_REQUESTS_LIMIT`
- `LLM_PRICING_JSON`
- 任务上下文与报告内容的字符预算配置
- 总结并发上限

这一步非常明显地提升了跨环境本地复现的一致性。

### 5.7 日志与请求中间件

`backend/src/main.py` 的 API 入口也有明显工程化变化。

原始版本使用的是 `loguru`。升级版则切换到了 Python 标准库 logging，并增加了：

- 统一的 `configure_logging()`
- `RequestContextFilter`
- 自动挂载 `request_id` 的 HTTP middleware
- 请求开始/结束日志
- 响应头中的 request ID 透传

这更贴近服务端工程实践，也和新的 request trace 模型更加自然地配合起来。

## 6. 前端源码变化

前端源码的变动非常克制，主要集中在 `frontend/src/App.vue`。

最重要的变化有：

- 页面顶部新增 metrics strip 区域
- 新增请求级指标展示：
  - 请求状态
  - 总耗时
  - cache hit/miss
  - token 数
  - estimated cost
  - 聚合成功率
- 新增对这些事件的处理：
  - `stage_started`
  - `stage_completed`
  - `fallback_triggered`
  - `degraded_response`
  - `metrics_snapshot`

因此，前端在不改变原有研究流程的前提下，更擅长向用户解释系统当前处于什么状态。

这是一个很典型的例子：保留原交互，只增强运行态透明度。

## 7. 测试、评测、部署与交付层升级

### 7.1 回归测试

原始示例没有与之对应的测试覆盖。

升级版补充了这些测试：

- 配置规范化与路径解析
- 同步执行正确性
- 搜索失败止损
- 总结失败止损
- API 响应兼容性
- SSE 契约
- metrics 快照接口
- 搜索缓存行为
- 离线 benchmark 输出

这说明项目的升级目标不只是“演示更漂亮”，而是真正提升可靠性。

### 7.2 离线 Benchmark 框架

`backend/evals/` 是一个非常实质性的 Phase 3 增量。

它提供了：

- benchmark schema
- JSON / JSONL 加载
- 批量运行器
- heuristic judge
- CLI 入口
- 结果文件输出

当前评分策略刻意做得比较轻量、可重复，主要指标包括：

- 是否生成报告
- degraded 标记
- section completeness
- keyword coverage
- citation count
- 总耗时
- estimated cost

这是一套很务实的第一版评测基础设施，已经明显超出了原始教程的范围。

### 7.3 Docker 与 CI

升级版新增了：

- `backend/Dockerfile`
- 根目录 `docker-compose.yml`
- `.github/workflows/ci.yml`

这些新增直接提升了复现性和验证能力：

- Docker 让别人更容易在较少本地假设的情况下跑起后端
- CI 明确给出了项目期望的验证路径：
  - backend lint
  - backend tests
  - wheel build check
  - frontend build

这代表了交付成熟度的明显提升。

### 7.4 项目级文档

原始项目只有一个很简短的后端 README。升级版则补齐了：

- 根目录 `README.md`
- `ENGINEERING.md`
- 扩展后的 backend README

这使得项目更容易交给同学、面试官、团队成员或评审来理解和复现。

## 8. 与 `Deepresearch_advanced.md` 的对齐度

从实现结果看，升级版和提示词的方向高度一致，尤其体现在这些方面：

- 配置管理
- 请求标识
- 日志与可观测性
- resilience / fallback 思路
- 保持 API 兼容
- 增加最小但有意义的测试

而且它已经超出了最初的 Phase 1 目标，继续落地了部分 Phase 2 和 Phase 3 的内容：

- metrics
- evals
- Docker
- CI
- 交付文档

不过，它并不是把提示词里的每一条 checklist 都逐字逐项实现出来。

下列能力在当前代码里没有看到非常明确的、显式的实现机制：

- 显式的 LLM timeout 与 retry/backoff 策略
- 显式的 fallback LLM provider 切换
- 显式的 search retry 策略
- 当所有 LLM 都失败时的规则化降级报告模板
- 严格 JSON schema 的日志输出

所以，更准确的描述应该是：

> 这个项目很好地遵循了提示词的工程化方向，但它是一版务实的工程升级，而不是严格逐条 checklist 全量完成的实现。

## 9. 总体判断

相较于原始的 chapter14 示例，升级版项目在五个维度上都有明显提升：

- 正确性
- 稳定性与止损能力
- 可观测性
- 可复现性
- 交付成熟度

如果只用一句话概括：

- 原始版本：教程风格的 deep research demo
- 升级版本：经过工程化加固、适合本地复现、调试、评测和展示的 deep research demo

## 10. 建议阅读顺序

如果你想最快验证本文结论，建议按这个顺序阅读：

1. `backend/src/agent.py`
2. `backend/src/main.py`
3. `backend/src/metrics.py`
4. `backend/src/config.py`
5. `backend/src/services/search.py`
6. `frontend/src/App.vue`
7. `backend/tests/`
8. `backend/evals/`
9. `README.md` 和 `ENGINEERING.md`
