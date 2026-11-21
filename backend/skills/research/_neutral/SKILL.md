---
pack_id: _neutral
schema_version: 1
version: 1
content_hash: "b93aae0d7f4ba1f6"
maturity: verified
status: active
supersedes: ""
title: 通用研究方法
topic_domain: general
applicable_when: 当研究主题无法明确归入任何专门领域，或没有任何专门包达到匹配阈值时的兜底方法
match_signals: []
negative_signals: []
min_match_score: 0.0
stage_scope:
  - planning
  - task_execution
  - reflection
  - review
  - report
quality_gates: {}
budget_knobs: {}
layout: {}
derived_from: ""
eval: {}
owner: research-team
created_at: 2026-01-01T00:00:00Z
updated_at: 2026-01-01T00:00:00Z
---

## planning
把主题拆成互补而非重叠的子问题，覆盖背景、现状、关键争议与不确定性。
每个子任务应能被独立检索与验证，避免一个任务同时追问过多方向。
优先安排能快速确认主干事实的任务，再安排用于补充细节与反证的任务。

## task_execution
先阅读已登记来源目录再决定是否补充检索，不要在信息已足够时继续扩张搜索。
围绕任务目标提炼结论，并为每个关键结论保留可核对的来源。
遇到相互矛盾的说法时，如实记录分歧，不要强行调和成单一结论。

## reflection
检查是否存在只被单一来源支撑、或完全没有被覆盖到的关键角度。
只有当缺口会实质影响结论可信度时才追加任务，避免为完备而无限扩张。
优先补齐反证与时效性，而不是重复已经充分的方向。

## review
确认每个重要结论都有来源支撑，且来源之间不是彼此转述的同一出处。
标记缺乏支撑、过期或与主流证据冲突的结论，交由报告阶段谨慎处理。

## report
按读者最关心的问题组织结构，先给结论再给依据。
对不确定或存在分歧的部分明确标注，不要用确定语气掩盖证据不足。
