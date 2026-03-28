# DeepResearch Agent Interview Guide

## 项目定位

- 这是一个面向开放互联网信息研究的 Agent 应用工程项目。
- 目标是把 `planning -> search -> summarization -> report` 做成可演示、可观测、可测试、可 benchmark 的闭环。
- 它故意不走知识库 RAG 主线，以便和金融研报 RAG 项目形成互补。

## 为什么需要 Planner

- 开放问题天然是多子问题的，需要先拆任务再检索。
- planner 输出结构化任务后，前端可以稳定展示任务卡片、状态和中间结果。
- 即使模型输出不稳定，当前仓库也有 table / numbered text / tool payload 的兜底解析。

## 为什么要做 SSE

- 研究型请求天然比普通问答更长，需要让用户看到阶段进度。
- SSE 能把 `status`、`todo_list`、`task_status`、`tool_call`、`metrics_snapshot` 逐步推到前端。
- 这类可视化过程对面试展示非常加分，因为它说明你考虑了长链路产品体验。

## 为什么要做 fallback / degraded

- 开放互联网研究经常会遇到搜索失败、总结失败、planner 输出异常。
- 当前实现不是一出错就 500，而是尽量保留部分任务结果，并把请求标成 `partial_success`。
- 这说明你不是只做 happy path，而是考虑了真实线上不稳定性。

## 为什么要做 metrics / perf / CI

- metrics 让你能看到 request trace、cache hit、token、estimated cost。
- perf smoke 和 regression baseline 让这个项目不止是“能跑”，而是“能测”。
- CI 让仓库更像一个工程项目，而不是一次性 demo。

## 和金融研报 RAG 项目的边界

- 金融研报 RAG 项目：重点讲文档解析、chunk、embedding、向量检索、rerank、引文溯源。
- DeepResearch Agent 项目：重点讲 Agent 编排、SSE 交互、工具调用、降级策略、可观测性、benchmark/CI。
- 两个项目一起放在简历里，能覆盖“RAG 系统”和“Agent 应用工程”两条线。
