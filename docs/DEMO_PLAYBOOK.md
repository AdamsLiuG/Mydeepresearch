# DeepResearch Agent Demo Playbook

这份 playbook 用来快速演示当前仓库的 Agent 工程能力，而不是知识库 RAG。

## 固定演示题目

| 场景 | 题目 | 建议亮点 |
| --- | --- | --- |
| 开放信息研究 | `探索多模态大模型在 2025 年的关键突破` | 展示 planner、多任务拆解、搜索汇总、最终报告 |
| Agent 工程实践 | `AI 搜索 Agent 在互联网信息研究中的工程化实践` | 展示工具调用、阶段事件、过程记录、metrics |
| 推理服务工程 | `开源大模型推理服务的部署、监控与成本控制` | 展示任务规划、引用来源、性能与成本叙事 |

## 5 分钟演示流程

1. 启动后端：
   `cd backend && cp .env.example .env && ./.venv/bin/python src/main.py`
2. 启动前端：
   `cd frontend && npm ci && npm run dev`
3. 打开前端，直接点击首页的固定演示场景按钮。
4. 演示重点按顺序讲：
   `任务规划卡片 -> 流程记录 -> 工具调用痕迹 -> 单任务总结 -> 最终报告 -> metrics`

## 30-60 秒录屏脚本

1. 首页点击一个固定演示场景。
2. 点击“开始研究”。
3. 等待右侧出现任务规划概览。
4. 滚动展示流程记录与工具调用。
5. 切到最终报告区域并停留 2-3 秒。

录屏时建议口播一句：
“这个项目不是知识库 RAG，而是一个联网研究型 Agent，重点展示任务规划、流式事件、降级和可观测性。”

## 保底演示模式

当你担心现场网络、模型或搜索 provider 波动时：

1. 将 `backend/.env` 里的 `BENCHMARK_STUB_ENABLED=True`
2. 重启后端
3. 用任意题目演示完整流程

这个模式会走 deterministic stub agent，适合展示：

- SSE 流式事件
- 任务状态切换
- metrics snapshot
- 前端整体交互闭环

## 降级 / fallback 截图建议

建议至少准备一张包含以下信息的截图：

- 流程记录中出现 `fallback_triggered` 或 `degraded_response`
- metrics 面板里能看到 `partial_success` 或 fallback 次数
- 右侧报告区域仍然有最终输出

## 面试表述建议

- 这是一个“联网研究型 Agent 应用工程项目”，不是文档知识库 RAG。
- 它强调的是流程编排、可观测性、降级策略和演示稳定性。
- 金融研报项目负责讲 RAG 检索链路；这个项目负责讲 Agent 产品和工程落地。
