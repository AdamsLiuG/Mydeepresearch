---
pack_id: tech-selection
schema_version: 1
version: 1
content_hash: "21237800e7b89153"
maturity: draft
status: active
supersedes: ""
title: 技术方案选型
topic_domain: tech_selection
applicable_when: 当研究主题涉及技术方案对比、组件选型、架构取舍或性能与工程权衡时
match_signals:
  - 技术选型
  - 方案对比
  - 架构
  - 组件
  - 框架
  - 性能
  - 权衡
  - 选型
negative_signals:
  - 政策
  - 法规
  - 合规
  - 竞品市场份额
min_match_score: 0.25
stage_scope:
  - planning
  - task_execution
  - reflection
  - review
  - report
quality_gates:
  review_min_domains_per_task: 2
budget_knobs:
  task_react_max_rounds: 3
layout: {}
retrieval_profile:
  preferred_backends:
    - tavily
    - duckduckgo
    - semanticscholar
  backend_weights:
    tavily: 1.15
    semanticscholar: 1.05
    duckduckgo: 0.90
  authoritative_hosts:
    - github.com
    - docs.python.org
    - kubernetes.io
    - postgresql.org
    - arxiv.org
    - ietf.org
  deprioritized_hosts:
    - csdn.net
    - jianshu.com
    - baidu.com
  prior_boost: 0.045
  prior_penalty: -0.025
verifiers:
  - numeric-audit
  - comparison-matrix
code_sandbox: true
composable_with:
  - policy-analysis
  - financial-research
  - competitor-scan
derived_from: ""
eval: {}
owner: research-team
created_at: 2026-01-01T00:00:00Z
updated_at: 2026-01-01T00:00:00Z
---

## planning
先明确选型的评价维度：功能覆盖、性能、可维护性、生态成熟度、迁移与运维成本。
为每个候选方案安排对称的任务，保证维度一致，避免只对某个方案深挖而忽略其他。
单列一个任务专门收集反面证据与已知局限，而不是只找支持某方案的材料。

## task_execution
用相同维度评估每个候选，记录来源与适用前提，注意结论往往依赖特定场景。
区分厂商自述与独立评测，性能类主张要留意测试条件是否可比。
遇到「银弹式」结论保持警惕，补充其代价与失败场景。

## reflection
检查是否所有候选都在同一组维度下被评估，是否遗漏了迁移成本或长期维护风险。
若某个关键维度只有单一来源，追加任务寻找独立佐证或反例。

## report
用维度对照的方式呈现候选差异，明确每项结论所依赖的前提与场景。
给出有条件的建议而非绝对最优，说明在什么约束下应选择哪种方案。

## review
确认对比在一致维度下进行，且性能与成熟度类主张有可比且独立的来源支撑。
标记仅凭厂商自述或单一场景得出的结论，提示其适用边界。
