---
pack_id: policy-analysis
schema_version: 1
version: 1
content_hash: "8691e21d27a37aa2"
maturity: draft
status: active
supersedes: ""
title: 政策与法规分析
topic_domain: policy_analysis
applicable_when: 当研究主题涉及政策、法规、监管要求、合规义务或制度适用范围时
match_signals:
  - 政策
  - 法规
  - 监管
  - 合规
  - 条例
  - 办法
  - 适用范围
  - 生效
negative_signals:
  - 技术选型
  - 架构
  - 竞品
  - 性能基准
min_match_score: 0.25
stage_scope:
  - planning
  - task_execution
  - reflection
  - review
  - report
quality_gates:
  review_min_sources_per_task: 3
  freshness_reference_days: 180
budget_knobs: {}
layout: {}
retrieval_profile:
  preferred_backends:
    - tavily
    - serpapi
    - duckduckgo
  backend_weights:
    tavily: 1.15
    serpapi: 1.05
    duckduckgo: 0.85
  authoritative_hosts:
    - .gov
    - gov.cn
    - europa.eu
    - nist.gov
    - cac.gov.cn
    - un.org
  deprioritized_hosts:
    - csdn.net
    - jianshu.com
    - quora.com
    - reddit.com
  prior_boost: 0.05
  prior_penalty: -0.03
verifiers:
  - timeline-audit
composable_with:
  - financial-research
  - tech-selection
  - competitor-scan
derived_from: ""
eval: {}
owner: research-team
created_at: 2026-01-01T00:00:00Z
updated_at: 2026-01-01T00:00:00Z
---

## planning
先定位发布主体与效力层级：区分法律、行政法规、部门规章与地方规范，不要混为一谈。
安排任务分别追问：适用主体与范围、生效与过渡安排、义务与罚则、以及最新修订状态。
把「原文条款」与「二手解读」拆成不同任务，避免只凭解读得出结论。

## task_execution
优先获取发布主体的权威原文，再用解读类来源补充背景，二者要能相互印证。
留意时效：条款可能已被修订或废止，确认所引内容仍然现行有效。
涉及适用范围与例外情形时逐条核对，不要用概括表述覆盖关键限定条件。

## reflection
检查是否遗漏了修订记录、配套细则或过渡期安排等容易被忽略的部分。
若关键义务或罚则只有二手来源支撑，应追加任务回到权威原文核实。

## review
确认对义务、范围与生效时间的表述都能追溯到权威原文，而非仅凭解读。
对已被修订或废止的条款要明确标记，避免以过期规定作为现行结论。

## report
按「适用对象—核心义务—例外与过渡—罚则」组织，先讲对读者最关键的适用判断。
对存在解释空间或尚未明确的条款，如实标注不确定性与信息来源层级。
