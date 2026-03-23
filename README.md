# HelloAgents DeepResearch Engineering Demo

基于 `helloagents-deepresearch` 示例做的二次工程化升级，目标不是重写框架，而是在保留原有 API 和前后端协作方式的前提下，把项目补齐为“可本地复现、可观察、可评测、可交付”的 Demo。

## 项目简介

- 后端：FastAPI + HelloAgents，多阶段执行 `planning -> search -> summarization -> report`
- 前端：Vue + Vite，消费 `/research/stream` SSE 事件展示研究进度
- 工程化增强：请求级追踪、进程内 metrics、搜索缓存、部分失败降级、离线 benchmark、Docker、CI

## 架构概览

```text
frontend (Vue)
  -> POST /research/stream
backend/src/main.py
  -> DeepResearchAgent
     -> PlanningService
     -> SearchTool / search cache
     -> SummarizationService
     -> ReportingService
  -> metrics_registry + RequestTrace
backend/evals
  -> benchmark loader
  -> heuristic judge
  -> batch runner
```

## 快速启动

### 1. 后端

在项目根目录执行：

```bash
cd backend
cp .env.example .env
/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.venv/bin/python src/main.py
```

启动后默认监听 `http://localhost:8000`。

### 2. 前端

```bash
cd frontend
npm ci
npm run dev
```

前端默认开发端口是 `5174`，会调用 `http://localhost:8000`。

## API

- `GET /healthz`
- `GET /metrics/json`
- `POST /research`
- `POST /research/stream`

## 环境变量说明

当前代码实际消费的核心变量如下，完整示例见 [backend/.env.example](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.env.example)。

| 分类 | 变量 |
| --- | --- |
| LLM | `LLM_PROVIDER`, `LLM_MODEL_ID`, `LLM_API_KEY`, `LLM_BASE_URL`, `LOCAL_LLM`, `LMSTUDIO_BASE_URL`, `OLLAMA_BASE_URL` |
| Search | `SEARCH_API`, `FETCH_FULL_PAGE`, `SEARCH_CACHE_ENABLED`, `SEARCH_CACHE_TTL_SECONDS` |
| Runtime | `HOST`, `PORT`, `LOG_LEVEL`, `CORS_ORIGINS` |
| Notes | `ENABLE_NOTES`, `NOTES_WORKSPACE` |
| Metrics | `METRICS_RECENT_REQUESTS_LIMIT`, `LLM_PRICING_JSON` |

## Fallback / Degrade 机制

- planner 没有产出任务时，会退化为单个兜底任务继续执行
- 单个任务的搜索失败或总结失败，不再直接让整条同步请求 500，而是标记任务为 `failed`
- 请求级结果会根据 fallback、degraded 原因和任务状态被标记为 `success` / `partial_success` / `failed`
- SSE 流会发出 `fallback_triggered`、`degraded_response`、`metrics_snapshot` 事件，方便前端展示和排障

## Metrics 说明

后端暴露 `GET /metrics/json`，包含：

- counters：请求总量、成功率、fallback 次数、搜索缓存命中等
- latencies：planning / search / summarization / report / total 的进程内耗时统计
- recent_requests：最近 N 次请求 trace
- estimated_cost：基于 `LLM_PRICING_JSON` 的静态成本估算

## Eval / Benchmark

Phase 3 新增了 `backend/evals/`，提供：

- `json` / `jsonl` benchmark 读取
- heuristic judge：`report_generated`、`degraded_flag`、`section_completeness`、`keyword_coverage`、`citation_count`、`total_latency_ms`、`estimated_cost`
- 批量运行脚本，默认会把结果输出到 `backend/evals/results/latest_results.json`

### benchmark 运行命令

```bash
cd backend
/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.venv/bin/python evals/run_benchmark.py --input evals/benchmarks/sample_benchmark.jsonl --output evals/results/sample_results.json
```

快速 smoke run：

```bash
cd backend
/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.venv/bin/python evals/run_benchmark.py --limit 2
```

> 说明：即使某些 case 因本地 LLM / Search 不可用而失败，runner 也会继续执行并产出结果文件，便于离线排查与比较。

## Docker

后端提供了 [backend/Dockerfile](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/Dockerfile) 和根目录 [docker-compose.yml](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/docker-compose.yml)。

### Docker 启动命令

```bash
cp backend/.env.example backend/.env
docker compose up --build
```

容器默认暴露 `8000`，并内置 `GET /healthz` 健康检查。环境变量通过 `backend/.env` 注入，任务笔记会落到挂载的 `backend/notes`。

## CI

CI 配置位于 [.github/workflows/ci.yml](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/.github/workflows/ci.yml)，包含：

- backend lint：`ruff check`
- backend test：`pytest`
- backend build check：`pip wheel ./backend --no-deps`
- frontend build check：`npm run build`

## 已知限制

- benchmark 目前是 deterministic / heuristic 版本，还没有接入 LLM-as-a-judge
- 真实 benchmark 数据依赖你本地可用的 LLM / Search Provider，默认结果应视为“待 benchmark 实测”
- 后端代码仍然保留了示例项目的顶层模块布局，虽然 Phase 3 已补齐 wheel 构建检查，但后续仍建议继续收敛为更标准的包结构
- Docker 当前只覆盖后端，前端仍建议在本地 Node 环境启动

## 更多文档

- 版本差异分析见 [CODEBASE_DIFF_ANALYSIS.md](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/CODEBASE_DIFF_ANALYSIS.md)
- 工程设计说明见 [ENGINEERING.md](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/ENGINEERING.md)
- 后端补充说明见 [backend/README.md](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/README.md)
