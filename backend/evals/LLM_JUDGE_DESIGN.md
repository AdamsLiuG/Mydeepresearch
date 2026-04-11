# Benchmark / Eval LLM Judge Design Draft

## 1. Background

当前仓库的离线评测链路已经具备基础骨架：

- `backend/evals/judges/base.py` 定义了可插拔 `Judge` 协议
- `backend/evals/judges/heuristic.py` 提供了第一版 deterministic judge
- `backend/evals/runner.py` 负责单 case / 批量 benchmark 执行
- `backend/evals/run_benchmark.py` 提供 benchmark CLI
- `backend/evals/run_http_suite.py` 提供 HTTP 级全链路验证

当前 heuristic judge 主要覆盖：

- `report_generated`
- `degraded_flag`
- `section_completeness`
- `keyword_coverage`
- `citation_count`
- `reference_match_rate`
- `grounded_bullet_ratio`
- `total_latency_ms`
- `estimated_cost`

这套方案的优点是稳定、低成本、适合 CI；缺点是更擅长判断“像不像一份合格报告”，不擅长判断“结论是否真的被证据支撑”。

因此，这里设计一套**用于 benchmark / eval 的 LLM-as-a-judge 二级评测方案**。它的定位是：

- 不替代现有 `HeuristicJudge`
- 不进入在线主链路
- 不等同于运行时 `ReviewService` 里的 optional LLM-assisted review
- 用于离线 benchmark、HTTP suite、结果复盘和回归对比

## 2. Problem Statement

当前评测层存在三个核心缺口：

1. 缺少事实性判断
   当前指标能看出报告结构和关键词覆盖，但无法判断关键结论是否被已有证据真正支持。

2. 缺少覆盖度语义判断
   现在的 `expected_keywords` / `expected_sections` 更偏格式和表面匹配，对“是否覆盖 benchmark 真正关心的研究维度”判断有限。

3. 缺少引用质量语义判断
   当前能统计 citation 数量、`source_id` 形式、reference section 存在性，但不一定能判断“引用是否相关、是否支撑结论、是否保守表述”。

## 3. Goals

本设计希望新增一个 `LLMJudge`，对单个 case 产物做二级评估，重点回答：

- 报告中的关键结论是否被证据支持
- benchmark case 关心的维度是否被覆盖
- 引用是否与结论匹配
- 对时效性问题是否给出足够新、足够明确的支撑
- 在证据不足时，模型是否保持保守表达

具体目标：

- 支持 `run_benchmark.py` 中按参数切换 judge
- 支持 `run_http_suite.py` 复用同一套 judge
- 保留现有 heuristic 结果，避免破坏已有结果口径
- 提供结构化 JSON 输出，便于落盘、汇总、对比
- 对 judge 自身失败具备容错，不影响整套 benchmark 继续执行

## 4. Non-Goals

本设计暂不包含以下内容：

- 不替换运行时 `ReviewService`
- 不把 LLM judge 接入在线 `review stage`
- 不把 LLM judge 作为 CI 强门禁
- 不尝试自动验证互联网事实真伪的“绝对正确性”
- 不在第一版解决人工标注、freshness gold label、跨评委一致性等完整评测体系问题

## 5. Clarification: LLM Judge vs Runtime Review Agent

这两个概念必须分开：

### 5.1 Runtime review agent

当前仓库已经有运行时审查链路：

- `backend/src/services/reviewer.py`
- `backend/src/agent.py`

它的职责是在**正式生成最终报告之前**，对任务摘要和证据做线上质量检查，输出：

- `review_issues`
- `review_status`
- `review_summary`
- `repair_candidates`

这属于产品运行时逻辑。

### 5.2 Eval-time LLM judge

本设计中的 LLM judge 属于**离线评测器**。它的输入是 benchmark case 和最终产物，职责是：

- 对 case 最终输出做事后打分
- 为 benchmark / regression 提供更强语义评估
- 辅助比较不同 prompt / model / search provider 的回归差异

一句话概括：

- runtime review agent 是“系统内部自检”
- eval-time LLM judge 是“系统外部裁判”

## 6. High-Level Architecture

建议新增以下组件：

- `backend/evals/judges/llm.py`
  实现 `LLMJudge`
- `backend/evals/judges/hybrid.py`
  实现 `HybridJudge`
- `backend/evals/prompts/llm_judge_prompt.md` 或在 `llm.py` 内嵌 prompt 模板
- `backend/evals/tests/test_llm_judge.py`
  覆盖 prompt 构建、JSON 解析、错误处理

整体结构：

1. benchmark runner 先执行 case，得到：
   - `report_markdown`
   - `todo_items`
   - `trace_snapshot`
2. 按配置选择 judge：
   - `heuristic`
   - `llm`
   - `hybrid`
3. judge 返回结构化 metrics
4. runner 持久化结果 JSON
5. summary 脚本聚合 heuristic 和 llm 两类指标

## 7. Proposed Judge Types

### 7.1 `HeuristicJudge`

保留现状，不改默认行为。

适用场景：

- 本地快速 smoke
- CI
- 低成本回归

### 7.2 `LLMJudge`

只输出 LLM 评测结果，重点看：

- 事实性
- 覆盖度
- 引用支撑
- freshness
- 保守性

适用场景：

- `real_local` benchmark
- 结果复盘
- 需要更接近人工判断的评测

### 7.3 `HybridJudge`

组合 heuristic + llm 结果。

推荐作为后续主力方案：

- heuristic 提供稳定基线
- llm 提供语义层补充

## 8. Inputs for LLM Judge

### 8.1 Existing inputs

`Judge.evaluate()` 已有输入：

- `case`
- `report_markdown`
- `todo_items`
- `trace_snapshot`

这套接口无需大改即可支持第一版 LLM judge。

### 8.2 Recommended derived inputs

在 `LLMJudge.evaluate()` 内部，建议把原始输入整理为更适合 prompt 的结构：

- case 基本信息
  - `id`
  - `topic`
  - `expected_sections`
  - `expected_keywords`
  - `freshness_sensitive`
  - `metadata`
- report 信息
  - 完整 `report_markdown`
  - 截断后的 `report_excerpt` 备用
- task 信息
  - `id`
  - `title`
  - `status`
  - `summary`
  - `sources_summary`
  - `claims`
  - `review_issues`
  - `review_status`
- trace 信息
  - `status`
  - `elapsed_ms`
  - `estimated_cost`
  - `degraded`
  - `degraded_reasons`
  - `fallback_triggered`
  - `fallback_reasons`

### 8.3 Optional future benchmark schema extensions

为了让 LLM judge 更稳定，建议后续给 `BenchmarkCase` 增加可选字段：

- `must_cover_angles: list[str]`
- `expected_claims: list[str]`
- `must_not_claim: list[str]`
- `preferred_source_types: list[str]`
- `freshness_expectation_days: int | None`
- `grading_notes: str | None`

第一版可以不改 schema，第二版再逐步引入。

## 9. Output Schema

建议 LLM judge 严格输出 JSON，字段如下：

```json
{
  "judge_status": "success",
  "judge_model": "gpt-5.4",
  "judge_version": "llm_judge_v1",
  "factuality_score": 0.82,
  "coverage_score": 0.75,
  "citation_grounding_score": 0.78,
  "freshness_score": 0.60,
  "conservativeness_score": 0.85,
  "overall_verdict": "warning",
  "reason": "报告总体覆盖较完整，但部分结论证据偏弱，且时效性支撑不足。",
  "findings": [
    {
      "severity": "high",
      "category": "unsupported_claim",
      "message": "关键结论对近期状态做出判断，但引用未明确支撑该时效性。"
    },
    {
      "severity": "medium",
      "category": "coverage_gap",
      "message": "缺少对 benchmark case 关注的部署成本维度的完整分析。"
    }
  ],
  "scoring_notes": {
    "strengths": [
      "报告结构清晰",
      "大部分结论具备 citation"
    ],
    "weaknesses": [
      "引用与结论匹配度不稳定",
      "freshness-sensitive case 中近期证据不足"
    ]
  }
}
```

### 9.1 Field definitions

- `judge_status`
  - `success | error | skipped`
- `factuality_score`
  - 0.0 到 1.0，判断结论是否被现有证据支持
- `coverage_score`
  - 0.0 到 1.0，判断是否覆盖 case 关键维度
- `citation_grounding_score`
  - 0.0 到 1.0，判断引用与结论是否匹配
- `freshness_score`
  - 0.0 到 1.0，仅在 freshness-sensitive case 更重要
- `conservativeness_score`
  - 0.0 到 1.0，证据不足时是否保持保守
- `overall_verdict`
  - `pass | warning | fail`
- `findings`
  - 结构化问题列表，用于复盘和人工抽查

## 10. Metric Semantics

### 10.1 Factuality

不是要求 judge 去上网验证事实，而是要求 judge 在**给定报告、task 结果、claims、citations、trace** 这些上下文里，判断报告的结论是否被当前证据充分支撑。

重点看：

- 是否存在明显超出证据范围的结论
- 是否把弱证据写成强断言
- 是否引用了不相关证据来支撑关键结论

### 10.2 Coverage

看 case 关注的关键维度是否被覆盖，不是简单 keyword hit。

重点看：

- 是否回答了 case 的核心问题
- 是否遗漏明显应该讨论的研究维度
- 是否只覆盖表层，不够成体系

### 10.3 Citation grounding

重点不是“有没有 citation”，而是：

- citation 是否真的支撑该句结论
- 结论是否与 citation 语义一致
- 是否存在“有引但不支撑”的情况

### 10.4 Freshness

仅当 case 明显是 freshness-sensitive 时重点评分。

重点看：

- 是否使用足够新的证据
- 是否明确说明证据时间范围
- 是否把历史资料误写成当前状态

### 10.5 Conservativeness

这是对研究型 agent 很重要的一项。

重点看：

- 证据不足时是否明确不确定性
- 是否保守表述限制条件
- degraded / partial_success 场景下是否过度自信

## 11. Prompt Design

### 11.1 System prompt

建议：

```text
You are a benchmark evaluator for a research agent.
Your task is to judge the final output quality of a single benchmark case.
You must only evaluate based on the provided benchmark case, report, task outputs, and trace data.
Do not assume facts not present in the supplied material.
Return one strict JSON object and nothing else.
```

### 11.2 Prompt constraints

必须强调：

- 只基于给定材料判断
- 不自行联网
- 不改写报告
- 不输出解释段落或 Markdown
- 严格输出 JSON
- 不把“写得像”误判成“事实正确”

### 11.3 Prompt sections

建议输入分块：

1. Case block
2. Expected dimensions block
3. Trace block
4. Task summary block
5. Final report block
6. Scoring rubric block
7. Output JSON schema block

### 11.4 Scoring rubric snippet

```text
Scoring guide:
- factuality_score: Are important claims supported by available evidence?
- coverage_score: Does the report cover the main dimensions implied by the benchmark case?
- citation_grounding_score: Are citations relevant and sufficient for the claims they support?
- freshness_score: For freshness-sensitive cases, does the report rely on recent enough evidence and state uncertainty properly?
- conservativeness_score: When evidence is weak, does the report remain careful rather than overclaim?
```

## 12. Implementation Plan

### Phase 1: Minimal viable `LLMJudge`

新增：

- `backend/evals/judges/llm.py`

能力：

- 调用一个固定 LLM
- 构建 judge prompt
- 解析 JSON
- 返回结构化 metrics
- judge 失败时返回 `judge_status=error`

### Phase 2: `HybridJudge`

新增：

- `backend/evals/judges/hybrid.py`

策略：

- 先跑 `HeuristicJudge`
- 再跑 `LLMJudge`
- 输出合并结果

合并建议：

- heuristic 指标保留原字段
- LLM 字段统一加前缀
  - `llm_factuality_score`
  - `llm_coverage_score`
  - `llm_citation_grounding_score`
  - `llm_freshness_score`
  - `llm_conservativeness_score`
  - `llm_overall_verdict`

### Phase 3: CLI integration

修改：

- `backend/evals/run_benchmark.py`
- `backend/evals/run_http_suite.py`

增加参数：

- `--judge heuristic`
- `--judge llm`
- `--judge hybrid`

可选附加参数：

- `--judge-model`
- `--judge-base-url`
- `--judge-api-key-env`
- `--judge-timeout-seconds`
- `--judge-cache-dir`

### Phase 4: Summary aggregation

修改：

- `backend/evals/runner.py`
- 相关 summary markdown 生成逻辑

目标：

- 汇总 heuristic 平均指标
- 汇总 llm 平均分
- 统计 `pass / warning / fail`
- 单独列出高风险 findings 数量

## 13. Configuration Proposal

建议新增独立配置，避免与 runtime 主模型强耦合：

- `EVAL_LLM_JUDGE_ENABLED`
- `EVAL_LLM_JUDGE_MODEL_ID`
- `EVAL_LLM_JUDGE_API_KEY`
- `EVAL_LLM_JUDGE_BASE_URL`
- `EVAL_LLM_JUDGE_TIMEOUT_SECONDS`
- `EVAL_LLM_JUDGE_TEMPERATURE`
- `EVAL_LLM_JUDGE_MAX_RETRIES`
- `EVAL_LLM_JUDGE_CACHE_DIR`
- `EVAL_LLM_JUDGE_VERSION`

建议策略：

- 若未显式配置，则默认关闭
- 若 CLI 显式选择 `--judge llm`，但环境变量缺失，则给出清晰错误
- `HybridJudge` 在 LLM judge 失败时仍保留 heuristic 结果

## 14. Caching Strategy

LLM judge 成本较高，建议引入本地缓存。

### 14.1 Cache key

建议由以下字段哈希生成：

- `case.id`
- `case.topic`
- `case.expected_sections`
- `case.expected_keywords`
- `report_markdown`
- `trace_snapshot.status`
- `judge_model`
- `judge_version`

### 14.2 Cache location

建议：

- `backend/evals/.cache/llm_judge/`

### 14.3 Cache semantics

- 命中缓存时直接复用 judge 结果
- prompt 版本变更后自动失效
- 模型变更后自动失效

## 15. Failure Policy

LLM judge 失败不应导致整个 benchmark suite 失败。

### 15.1 Allowed failures

- LLM 请求失败
- 超时
- 非 JSON 输出
- JSON parse 失败
- 分数越界

### 15.2 Fallback behavior

返回：

```json
{
  "judge_status": "error",
  "judge_error": "timeout",
  "overall_verdict": "warning"
}
```

行为：

- `run_benchmark.py` 继续执行后续 case
- `HybridJudge` 保留 heuristic 结果
- summary 中单独统计 `judge_error_cases`

## 16. Result File Shape

建议在结果文件中保留更清晰的分层，而不是把所有字段平铺混在一起。

推荐结构：

```json
{
  "id": "case_x",
  "topic": "...",
  "metrics": {
    "heuristic": {
      "report_generated": true,
      "section_completeness": 0.8
    },
    "llm": {
      "judge_status": "success",
      "factuality_score": 0.82,
      "coverage_score": 0.75,
      "overall_verdict": "warning"
    },
    "combined": {
      "report_generated": true,
      "section_completeness": 0.8,
      "llm_factuality_score": 0.82,
      "llm_coverage_score": 0.75,
      "llm_overall_verdict": "warning"
    }
  }
}
```

若考虑最小改动，也可以先让 `HybridJudge.evaluate()` 直接输出兼容旧结构的扁平字段。

## 17. Interaction with Existing HTTP Suite

`run_http_suite.py` 当前会检查：

- HTTP 200
- SSE 事件完整性
- 最终报告是否存在
- 启发式 judge 指标

接入 LLM judge 后建议保持两层结论：

1. 工程链路是否通过
   - HTTP / stream / final_report / done / error event
2. 语义质量是否达标
   - factuality / coverage / citation grounding / freshness

这样可以避免把“链路成功”和“答案质量高”混成一个指标。

## 18. Recommended Rollout Strategy

### Step 1

先实现 `LLMJudge`，但只支持 `run_benchmark.py --judge llm`

### Step 2

实现 `HybridJudge`，支持：

- `run_benchmark.py --judge hybrid`
- `run_http_suite.py --judge hybrid`

### Step 3

为 `full_system_12cases.jsonl` 增加更丰富的 benchmark metadata

### Step 4

让 summary markdown 同时展示：

- heuristic 均值
- llm 均值
- `pass / warning / fail`
- judge error 数

### Step 5

人工抽样校准 5-10 个 case，检查 judge 是否过严或过松

## 19. Risks

### 19.1 Cost risk

LLM judge 会显著增加 benchmark 成本和耗时。

缓解：

- 结果缓存
- 只在 `real_local` 或 nightly regression 启用
- 对大报告做结构化裁剪

### 19.2 Stability risk

同一份报告多次评分可能波动。

缓解：

- 固定模型
- 温度设为 0
- 固定 prompt version
- 输出严格 JSON

### 19.3 False confidence risk

LLM judge 不是事实判官，仍然可能把“写得像”误判为“真的对”。

缓解：

- 明确其定位是二级评估而非绝对真理
- 保留 heuristic
- 对关键回归做人工 spot-check

### 19.4 Coupling risk

若直接复用 runtime 主模型，可能导致 benchmark 成本不可控。

缓解：

- 给 eval judge 独立配置
- 优先选择稳定、较便宜的 judge 模型

## 20. Test Plan

### 20.1 Unit tests

- prompt 构建包含必要字段
- 非法 JSON 能正确报错
- 合法 JSON 能正确 parse
- score 越界时能被裁剪或拒绝
- cache key 对 prompt version / model 变更敏感

### 20.2 Integration tests

- 用 mock judge response 跑单 case benchmark
- 用 mock judge response 跑 HTTP suite
- 验证结果文件格式稳定

### 20.3 Manual calibration

人工挑选若干 case：

- 高质量样本
- 明显漏维度样本
- citation 弱支撑样本
- freshness-sensitive 样本

对比 judge 输出是否符合直觉。

## 21. Suggested Implementation Order

推荐按以下顺序实施：

1. `llm.py`
2. `test_llm_judge.py`
3. `run_benchmark.py --judge`
4. `hybrid.py`
5. `run_http_suite.py --judge`
6. benchmark schema 扩展
7. summary markdown 扩展

## 22. Example Task Breakdown

如果按工程任务拆分，可以拆成：

1. 定义 `LLMJudge` 输出 schema 与 parser
2. 实现 prompt builder
3. 实现 LLM 调用与缓存
4. 实现 `HybridJudge`
5. 扩展 benchmark CLI
6. 扩展 HTTP suite
7. 增加测试与样例结果

## 23. Decision Summary

最终建议如下：

- 保留 `HeuristicJudge` 作为基础评测层
- 新增 `LLMJudge` 作为离线 benchmark / eval 的二级语义评估
- 推荐以 `HybridJudge` 作为长期方向
- 不把 LLM judge 与 runtime review agent 混为一谈
- 第一阶段不上 CI 强门禁，只用于 `real_local` benchmark 和结果复盘

这套方案最适合当前仓库的原因是：

- 已经有 `Judge` 协议，扩展成本低
- 已经有 benchmark runner，接入路径明确
- 当前最大的缺口正是 heuristic 无法覆盖的事实性、覆盖度和引用质量语义判断
- 与现有 `ReviewService` 职责互补，而不是冲突
