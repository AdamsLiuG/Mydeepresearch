# HelloAgents Deep Researcher Backend

后端服务基于 FastAPI，保留现有四个接口：

- `GET /healthz`
- `POST /research`
- `POST /research/stream`
- `GET /metrics/json`

本轮工程化升级保持了原有接口语义和前端 SSE 协作方式，补充了以下能力：

- 自动加载 [`.env`](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.env)
- 请求级 `X-Request-ID` 追踪
- 统一标准库日志输出
- 将相对 `NOTES_WORKSPACE` 解析到后端目录下，减少启动目录差异带来的不确定性
- 最小回归测试，覆盖健康检查、同步研究接口、流式研究接口和配置解析
- Phase 2 新增进程内 metrics、request/task trace、搜索缓存、成本估算接口

## 运行环境

- Python 3.12
- 推荐使用仓库内现有虚拟环境：`/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.venv`

## 启动后端

在后端目录执行：

```bash
/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.venv/bin/python src/main.py
```

如果你使用 `uv`：

```bash
uv run python src/main.py
```

默认会读取 [`.env`](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.env) 配置；可以参考 [`.env.example`](/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.env.example) 里的字段，并使用其中的 `HOST`、`PORT`、`LOG_LEVEL`、`CORS_ORIGINS` 等参数。

## Phase 2 观测点

- `/metrics/json` 返回进程内聚合统计与最近请求 trace
- SSE 新增事件：`stage_started`、`stage_completed`、`fallback_triggered`、`degraded_response`、`metrics_snapshot`
- 搜索缓存默认开启，命中率会体现在 `/metrics/json`
- token 默认基于 prompt/response 文本长度估算；后续可继续接入真实 usage
- `estimated_cost` 基于 `LLM_PRICING_JSON` 静态单价配置估算；未配置时默认是 `0`

## 运行测试

```bash
/media/main/hjz/agent/deepresearch/helloagents-deepresearch/backend/.venv/bin/python -m unittest discover -s tests -v
```

## 手工验证

1. 启动后端服务。
2. 访问 `http://localhost:<PORT>/healthz`；默认 `PORT=8000`，确认返回 `{"status":"ok"}`。
3. 访问 `http://localhost:<PORT>/metrics/json`，确认能看到 counters、latencies、recent_requests。
4. 用前端发起一次研究，或直接调用 `POST /research/stream`。
5. 检查响应头里是否包含 `X-Request-ID`，并确认前端流程区出现阶段事件和 metrics 卡片。
6. 再次用完全相同的主题和搜索引擎发起请求，然后重新访问 `/metrics/json`，确认 `cache_hit_total` 增长。
7. 让 planner 返回空任务列表，确认会出现 `fallback_triggered`，并看到 `request_partial_success_total` 或 `fallback_trigger_total` 增长。
8. 若配置了 `LLM_PRICING_JSON`，确认 `/metrics/json` 中 `estimated_cost` 和每次请求的 `total_tokens` 非零。
