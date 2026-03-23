#### 总指令

你现在扮演这条项目线的资深工程师，目标是在现有仓库基础上，对 /media/main/hjz/agent/deepresearch/helloagents-deepresearch示例做“最小侵入式工程化升级”。

你的工作原则不是重写，而是：

1. 保留现有主流程、接口语义和前后端协作方式
2. 优先提升稳定性、可观测性、可复现性
3. 每一步都必须可运行、可验证、可回退
4. 先读代码，再改代码；不要凭空假设函数名和文件内容

工作方式要求：

- 先扫描仓库，确认真实目录、入口、依赖、现有接口
- 再提出执行计划
- 然后按阶段实施
- 每一阶段结束时都要给出可运行结果
- 如果发现现有实现与预期不一致，优先适配现有代码，而不是强行重构
- 如果某个改动风险太高，就缩小范围，先交付低风险版本
- 不要一次性大改多个横切面；先稳住后端，再补指标，再补评测和交付

强约束：

- 不允许破坏现有 API 路径和前端基本交互
- 不允许把项目改造成完全不同的框架
- 不允许只给概念建议而不落代码
- 不允许引入大量不必要依赖
- 不允许为了“优雅”而牺牲可运行性
- 不允许省略测试和运行说明

每次输出必须包含：

1. 你读到的当前仓库真实情况
2. 本轮改动目标
3. 修改文件列表
4. 每个文件的关键改动
5. 运行命令
6. 测试命令
7. 手工验证步骤
8. 风险、已知限制、下一步建议

如果你遇到不确定点，优先通过读取仓库确认；只有在确实无法确认时，才做最保守假设并明确写出假设。



#### Phase 1：稳定性与止损优先

先做 Phase 1，不要提前做 Phase 2/3。

【Phase 1 目标】
把当前 deepresearch 示例从“教程可演示”提升到“本地可稳定运行、失败可止损、日志可排障”。

【业务视角】
这个系统是一个长链路研究任务，不是简单聊天接口。当前最重要的问题不是多加功能，而是：

- 配置不清晰
- 失败时容易整条链路中断
- 排障信息不足
- 运行和复现门槛高

所以这一阶段只聚焦四件事：

1. 配置管理
2. 结构化日志
3. LLM/Search 的 resilience
4. 最小测试与 smoke check

【明确非目标】
这一阶段不要做：

- 大规模重构
- 前端大改
- Prometheus/Grafana 等完整监控栈
- 复杂 benchmark 框架
- 数据库引入
- 消息队列引入

【你要先做的事】
先扫描仓库并确认：

- 后端真实入口文件
- src layout 是否正确
- pyproject.toml 是否可用
- 当前 config.py 怎么工作
- planner/summarizer/reporter/search 的真实调用关系
- /research/stream 的实现细节
- 现有异常处理和日志位置

【交付要求】

A. 配置管理
目标：把配置从“能跑就行”提升到“可读、可校验、可扩展”。

请实现：

- 统一 settings 层
- 环境变量分组：app / llm / search / fallback / observability
- 启动时校验关键配置
- 缺失关键配置时输出人类可读错误，不要栈追踪糊脸
- 提供 .env.example
- 保留对现有 config.py 的兼容，必要时在内部重构但不要破坏现有引用方式

至少支持：

- APP_ENV
- APP_LOG_LEVEL
- LLM_PROVIDER
- LLM_MODEL_ID
- LLM_API_KEY
- LLM_BASE_URL
- LLM_TIMEOUT_SECONDS
- LLM_MAX_RETRIES
- FALLBACK_LLM_PROVIDER
- FALLBACK_LLM_MODEL_ID
- FALLBACK_LLM_API_KEY
- FALLBACK_LLM_BASE_URL
- SEARCH_PROVIDER
- SEARCH_API_KEY
- SEARCH_TIMEOUT_SECONDS
- SEARCH_MAX_RETRIES
- ENABLE_RULE_BASED_DEGRADE
- WORKSPACE_DIR

B. 结构化日志
目标：让我能从一条 request 的日志看懂整条研究链路。

请实现：

- 统一 logger，替代散落的 print
- 为每个研究请求生成 request_id
- 如果已有研究任务维度，也生成 task_id / trace_id
- 日志字段至少包括：
  - timestamp
  - level
  - request_id
  - stage
  - event
  - provider
  - model_or_engine
  - elapsed_ms
  - error_type
- 关键事件必须打点：
  - request_received
  - planning_started / completed / failed
  - search_started / completed / failed
  - summarization_started / completed / failed
  - report_started / completed / failed
  - fallback_triggered
  - degraded_response_emitted
  - request_finished

C. Resilience / fallback / degrade
目标：单个依赖失败时，不要轻易让整条研究请求 500。

请实现：

- LLM 调用增加 timeout
- LLM 调用增加 retry with backoff
- 主模型失败时切换 fallback 模型
- Search 调用增加 timeout 和 retry
- Search 失败时记录结构化错误，并允许后续流程继续
- 子任务失败时，不要直接让整份报告失败；至少允许部分成功
- 当信息不足时，返回一个“最小可交付报告”，而不是抛 500
- 当 ENABLE_RULE_BASED_DEGRADE=true 且 LLM 全失败时，输出模板化降级报告，至少包含：
  - 研究主题
  - 已完成子任务
  - 失败子任务
  - 已收集到的来源/证据
  - 不确定性说明
  - 建议下一步

D. 最小测试
目标：确保这不是只在理想路径可运行。

至少补以下测试：

- 配置加载成功
- 配置缺失时报错清晰
- 主模型失败时 fallback 生效
- 搜索失败不导致整个请求直接崩溃
- 规则降级报告可生成
- 至少一个 smoke test 可以跑主入口或核心流程

【代码组织要求】
优先新增轻量模块，不要乱改所有文件。
如果合适，可以新增：

- src/core/settings.py
- src/core/logging.py
- src/core/errors.py
- src/core/retry.py
- src/core/fallback.py
- src/core/ids.py

但是否新增，以当前仓库最自然的方式为准。

【验收标准】
交付时必须满足：

- 本地可安装、可启动
- 缺关键环境变量时有清晰错误
- 任意一次研究请求有 request_id 可追踪
- 主模型失败时能自动切换备用模型
- 搜索失败时不会立刻把整条请求打崩
- 最差情况下也能产出最小报告
- pytest 至少有基础覆盖
- README 有最小运行说明

【输出格式】
按下面格式输出，不要省略：

1. 当前仓库扫描结论
2. 本轮实施方案
3. 修改文件列表
4. 分文件改动说明
5. 运行命令
6. 测试命令
7. 手工验证 checklist
8. 已知限制



#### Phase 2：可观测性与成本闭环

现在开始 Phase 2。前提：沿用上一阶段代码，不要推翻重来。

【Phase 2 目标】
把系统从“能排障”提升到“能量化分析”：知道哪里慢、哪里贵、哪里容易失败。

【业务视角】
这是一个研究型 Agent，不是普通问答接口。端到端耗时长、外部依赖多、成本波动大，所以必须建立最小可用的观测体系。

【明确非目标】
这一阶段不要做：

- 引入完整观测平台
- 引入外部数据库
- 大规模前端改版
- 复杂权限系统

【你要做的事】

A. Trace 与阶段耗时

- 为每次 research request 建立完整 trace
- 为每个子任务建立 task/span 级别追踪
- 记录 planning/search/summarization/report 四阶段的：
  - start_time
  - end_time
  - elapsed_ms
  - status
  - error
- 尽量复用 Phase 1 的 request_id/logging 体系

B. Metrics
实现一个轻量 metrics 模块，先做进程内聚合即可，不要求上来就接入外部系统。

至少统计：

- request_total
- request_success_total
- request_partial_success_total
- request_failed_total
- fallback_trigger_total
- llm_call_total / llm_success_total / llm_failed_total
- search_call_total / search_success_total / search_failed_total
- cache_hit_total / cache_miss_total
- planning_latency_ms
- search_latency_ms
- summarization_latency_ms
- report_latency_ms
- total_latency_ms

如果 provider 可拿到 token 信息，则增加：

- prompt_tokens
- completion_tokens
- total_tokens

请实现成本估算：

- 支持按 provider + model 的静态单价配置
- 输出 estimated_cost
- 如果真实 token 拿不到，也要保留扩展接口，不要把设计写死

C. Search 缓存

- 给 search service 增加简单缓存
- 先实现内存缓存或 workspace 文件缓存即可
- cache key 至少基于 query + provider
- 避免相同查询重复调用外部搜索
- 缓存命中也要记录指标

D. Metrics 暴露
新增一个轻量只读接口，例如：

- /metrics/json
  或
- /admin/metrics

返回最近聚合统计即可。
重点是让开发者能看到：

- 请求成功率
- 平均耗时
- fallback 次数
- 搜索缓存命中率
- 估算 token 与成本

E. SSE 事件增强
在不破坏现有前端交互的前提下，增强 SSE 事件内容：

- stage_started
- stage_completed
- fallback_triggered
- degraded_response
- metrics_snapshot

如果前端已有日志区域或进度区域，把这些事件展示出来；不要做大规模 UI 重写。

F. 测试
至少补充：

- metrics 累加正确
- fallback 时 metrics 正确更新
- cache hit 生效
- SSE 能发出新增事件

【验收标准】

- 一次研究任务结束后，能看到分阶段耗时
- 能看到 request 成功/失败/部分成功统计
- 能看到 fallback 触发次数
- 能看到 search cache hit/miss
- 如果可得，能看到 token 和估算 cost
- 前端或接口能展示这些关键指标

【交付要求】
输出时请明确：

1. 这轮新增了哪些 metrics
2. 各指标在哪些代码路径更新
3. 如何人工制造 fallback / cache hit 来验证
4. 哪些 token/cost 是真实值，哪些是估算值



#### Phase 3：评测、Docker、CI、文档交付

现在开始 Phase 3。目标是把项目补齐为“可复现、可评测、可交付”的工程化 Demo。

【Phase 3 目标】

1. 给项目加最小可用 benchmark/eval
2. 提供 Docker 化运行方式
3. 提供 CI
4. 提供一份面向他人的工程文档
5. 最后输出可写进简历的项目总结

【业务视角】
当前项目的价值不只是能跑，而是：

- 别人能复现
- 我们能比较优化前后差异
- 能以工程材料形式交给面试官或同学

【明确非目标】
这一阶段不要做：

- 复杂在线评测平台
- 复杂多机部署
- 大而全运维系统
- 与原项目完全分叉的结构重组

【你要做的事】

A. Evaluation / Benchmark
新增轻量 eval 模块，例如：

- backend/evals/

要求：

- 支持读取 benchmark json/jsonl
- 每条 benchmark 至少包含：
  - id
  - topic
  - expected_keywords
  - expected_sections
  - freshness_sensitive
- 提供一个离线脚本，能够批量运行研究流程并输出结果 json
- 初版指标至少包含：
  - report_generated
  - degraded_flag
  - section_completeness
  - keyword_coverage
  - citation_count
  - total_latency_ms
  - estimated_cost

注意：

- 先做 deterministic/heuristic 评测，不要求上来就接 LLM-as-a-judge
- 代码结构要为以后扩展 judge 接口留口子

B. 示例 benchmark

- 提供 10 到 20 条样例
- 主题覆盖：
  - 概念解释
  - 对比分析
  - 时效性话题
  - 多来源交叉验证

C. Docker

- 为后端提供 Dockerfile
- 如有必要提供 docker-compose.yml
- 目标是让他人最少步骤启动
- 提供 health check
- 明确环境变量注入方式

D. CI

- 新增 GitHub Actions
- 至少包括：
  - lint
  - pytest
  - 后端构建检查
- 如果前端已有基本构建命令，也加一个 build check

E. 文档
更新 README，并补一份工程文档，例如 ENGINEERING.md。

README 至少包含：

- 项目简介
- 架构概览
- 快速启动
- 环境变量说明
- fallback 机制说明
- metrics 说明
- eval 说明
- Docker 说明
- 已知限制

ENGINEERING.md 至少包含：

- 为什么做这些工程化升级
- resilience 设计
- observability 设计
- eval 设计
- 已知风险与后续路线

F. 简历项目总结
最后请输出一段可直接写进简历的项目描述，内容要包括：

- 基于什么项目做的二次升级
- 做了哪些稳定性设计
- 做了哪些可观测性设计
- 做了哪些评测与交付能力建设
- 如果已有数据，给出可量化结果
- 如果暂时没有真实数据，明确标注为“待 benchmark 实测”

【验收标准】

- benchmark 能跑并产出结果文件
- Docker 可启动
- CI 可执行基础检查
- README 足够让别人复现
- 工程文档能解释为什么这样设计
- 能输出一段面向简历的精炼总结

【输出格式】
保持与前两阶段一致，并额外补充：

1. benchmark 运行命令
2. Docker 启动命令
3. CI 文件说明
4. README / ENGINEERING.md 摘要
5. 简历版项目描述



#### 代码审查模式
现在不要继续开发新功能，请切换为“严格代码审查模式”。

请你审查刚才的改动，重点检查：

1. 是否破坏了现有 API 或前端依赖
2. 是否引入了不必要的复杂度
3. 是否有异常分支未覆盖
4. 是否有日志字段不一致
5. 是否有配置项命名混乱
6. 是否有 fallback 逻辑写死或不可扩展
7. 是否有测试只测 happy path
8. 是否有 README 与实际命令不一致
9. 是否有依赖未声明
10. 是否有 import path / src layout 风险

输出要求：

- 先列出高风险问题
- 再列出中风险问题
- 再列出低风险问题
- 对每个问题给出修复建议
- 如果问题明确且低风险，请直接顺手修复
- 如果问题可能破坏现有行为，请先说明影响再修复



