# Engineering Notes

## 为什么做这些工程化升级

这个项目原本更接近“教程级可演示示例”，Phase 1/2/3 的工程化目标是把它升级成一份可以交给同学、面试官或团队成员复现的材料。这里的重点不是引入更复杂的框架，而是在保留原有主流程的前提下补齐稳定性、观测性、评测与交付能力。

## Resilience 设计

当前链路是典型的长路径 agent workflow：`planning -> search -> summarization -> report`。这类系统的核心风险不是单个函数报错，而是中间任一依赖异常就把整条请求打断。

本轮设计重点包括：

- planner 空结果时退化为单个 fallback task，避免请求直接终止
- 单任务搜索失败和总结失败被视为可恢复错误，任务状态会标成 `failed`，但请求仍可继续生成最终报告
- 请求整体状态分为 `success` / `partial_success` / `failed`，让下游更容易解释“结果可用性”
- benchmark runner 也采用“逐 case 记录错误、继续执行”的策略，这样即使外部依赖暂时不可用，仍然能沉淀结果文件

## Observability 设计

可观测性的目标是让一次研究请求能够被追踪、比较、复盘。

当前项目已有的关键观测点：

- HTTP 层统一打 `X-Request-ID`
- `RequestTrace` 聚合每个请求的阶段、任务状态、缓存命中、token 与成本估算
- `/metrics/json` 暴露 counters、latencies 和 recent request traces
- SSE 流把 `stage_started`、`stage_completed`、`fallback_triggered`、`degraded_response`、`metrics_snapshot` 实时暴露给前端

这套设计的优点是实现轻量、无外部依赖，缺点是日志字段还不是严格 JSON schema，后续如果需要接入 ELK / Loki / Datadog，需要再统一日志字段格式。

## Eval 设计

Phase 3 的 eval 明确选择了最小可用路线，而不是一步到位上 LLM-as-a-judge。

### 当前实现

- `backend/evals/schema.py`：benchmark case schema
- `backend/evals/loader.py`：支持 `json` / `jsonl`
- `backend/evals/judges/base.py`：定义 judge 协议，给后续 judge 扩展留口子
- `backend/evals/judges/heuristic.py`：基于文本匹配的 deterministic 指标
- `backend/evals/runner.py`：批量运行 case，输出汇总结果
- `backend/evals/run_benchmark.py`：CLI 入口

### 当前指标

- `report_generated`
- `degraded_flag`
- `section_completeness`
- `keyword_coverage`
- `citation_count`
- `total_latency_ms`
- `estimated_cost`

### 为什么这样做

- heuristic judge 对依赖要求低，适合本地开发和 CI smoke check
- 结果可重复，适合作为优化前后的第一版比较基线
- 先把 benchmark 数据结构和 runner 铺好，后续才容易叠加 LLM judge、人工标注或更复杂的 freshness 检查

## Docker / CI 设计

Docker 和 CI 的目标是让“别人能跑”和“改动能被验证”。

### Docker

- `backend/Dockerfile` 只覆盖后端，保持镜像职责清晰
- 使用项目内 `pyproject.toml` 安装依赖，避免维护额外 requirements 文件
- 内置 `healthz` 健康检查
- `docker-compose.yml` 使用 `backend/.env` 注入变量，并把 `backend/notes` 挂载出来

### CI

- backend：ruff + pytest + wheel build check
- frontend：npm build

这里刻意没有引入复杂 matrix、缓存预热或多环境测试，因为当前目标是让 Demo 稳定、清晰、最少维护成本。

## 已知风险与后续路线

### 当前风险

- 日志目前仍偏“结构化 message”，不是严格 JSON log schema
- benchmark 还没有 freshness-aware scoring，也没有事实正确性判定
- backend 仍保留了示例项目的顶层模块布局，长期看最好继续收敛成标准 package layout
- Docker 目前不覆盖前端和完整端到端联调

### 后续可选路线

1. 接入 LLM-as-a-judge，对事实性、覆盖度和引用质量做二级评估
2. 把日志统一成 JSON 格式，并对齐 `request_id / stage / provider / elapsed_ms / error_type`
3. 增加 benchmark 基线快照，对比不同 prompt / model / search provider 的回归
4. 补前端容器化与 e2e smoke test，形成完整交付包
