# DeepResearch Agent Backend

后端是一个面向开放互联网研究任务的 FastAPI Agent 服务，核心目标是把长链路研究请求做成：

- 可流式展示
- 可降级恢复
- 可观测
- 可 benchmark

## 对外接口

- `GET /healthz`
- `POST /research`
- `POST /research/stream`
- `GET /metrics/json`

## 核心能力

- 自动加载 [`.env`](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.env)
- 请求级 `X-Request-ID` 追踪
- request / task trace
- 任务预算控制，避免 planner 一次生成过多任务拖慢演示
- 泛化 query 自动重写，降低短 query 空检索概率
- 搜索工具调用超时保护，避免外部 provider 长时间卡住任务
- 搜索工具失败重试参数化，可按本地环境调节容错策略
- 搜索 exact / semantic cache
- fallback / degraded 响应标记
- 估算 token / cost 指标
- benchmark stub agent，用于稳定演示和 CI perf

## 启动方式

在后端目录执行：

```bash
cp .env.example .env
/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.venv/bin/python src/main.py
```

如果你使用 `uv`：

```bash
uv run python src/main.py
```

实际配置入口是 [`.env`](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.env)；[`.env.example`](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.env.example) 只是模板。

开发模式的热重载只监听 [`src`](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/src) 源码目录，避免请求执行时写入 `notes / .state / .memory / .cache` 触发自重启；如果你修改了 [`.env`](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.env)，请手动重启后端进程。

和这两个轻量 Agent 能力直接相关的配置项：

- `MAX_AGENT_TASKS`：限制单次请求最多执行多少个任务
- `REQUEST_REFLECTION_ENABLED`：是否在首轮任务后做一次请求级覆盖反思
- `REFLECTION_MAX_ADDITIONAL_TASKS`：反思命中缺口后最多再补多少个任务
- `TASK_QUERY_REWRITE_ENABLED`：对过短或过泛的任务 query 自动拼接 `topic + title + intent`
- `SEARCH_TOOL_TIMEOUT_SECONDS`：限制单次搜索工具调用最长等待时间
- `SEARCH_TOOL_RETRY_ATTEMPTS`：搜索工具失败或超时后的重试次数
- `SEARCH_TOOL_RETRY_BACKOFF_SECONDS`：两次重试之间的固定等待时间
- `SEARCH_API`：搜索后端，可选 `duckduckgo / tavily / perplexity / searxng / semanticscholar / advanced`
- `SEMANTIC_CACHE_WARMUP_ENABLED`：是否在后端启动时预热 semantic cache embedding 模型，降低首轮搜索冷启动超时概率
- `SEMANTIC_SCHOLAR_API_KEY`：当 `SEARCH_API=semanticscholar` 时建议配置，避免共享限流

## 重点观测点

- `/metrics/json` 返回进程内 counters、latencies、recent request trace
- SSE 事件覆盖 `stage_started`、`stage_completed`、`fallback_triggered`、`degraded_response`、`metrics_snapshot`
- 搜索缓存会区分 exact hit、semantic hit、miss
- `estimated_cost` 基于 `LLM_PRICING_JSON` 估算；未配置时默认是 `0`

## 演示保底模式

如果你需要稳定展示流程，而不是依赖现场网络 / provider：

```bash
BENCHMARK_STUB_ENABLED=True
BENCHMARK_PROFILE=stub
```

这个模式适合演示：

- SSE 事件流
- 任务状态切换
- metrics snapshot
- 前端流程可视化

## 运行测试

```bash
/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.venv/bin/python -m unittest discover -s tests -v
```

## 最短验证路径

1. 启动后端。
2. 访问 `http://localhost:8000/healthz`。
3. 访问 `http://localhost:8000/metrics/json`。
4. 用前端或直接调用 `POST /research/stream` 发起一个固定 demo 题目。
5. 检查是否出现任务规划、阶段事件、工具调用、最终报告和 metrics 卡片。
