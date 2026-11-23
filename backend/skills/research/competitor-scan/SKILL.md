---
pack_id: competitor-scan
schema_version: 1
version: 1
content_hash: "d1964de9f5ad21c4"
maturity: draft
status: active
supersedes: ""
title: 竞品与市场扫描
topic_domain: competitor_scan
applicable_when: 当研究主题涉及竞品分析、市场格局、厂商动态或商业策略对比时
match_signals:
  - 竞品
  - 市场
  - 厂商
  - 份额
  - 商业模式
  - 定价
  - 格局
  - 对手
negative_signals:
  - 法规原文
  - 学术综述
  - 方法论
  - 架构选型
min_match_score: 0.25
stage_scope:
  - planning
  - task_execution
  - reflection
  - review
  - report
quality_gates:
  review_min_domains_per_task: 2
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
    duckduckgo: 0.90
  authoritative_hosts:
    - techcrunch.com
    - bloomberg.com
    - reuters.com
    - gartner.com
    - github.com
  deprioritized_hosts:
    - csdn.net
    - jianshu.com
    - sohu.com
  prior_boost: 0.04
  prior_penalty: -0.025
verifiers:
  - numeric-audit
  - comparison-matrix
composable_with:
  - financial-research
  - tech-selection
  - policy-analysis
derived_from: ""
eval: {}
owner: research-team
created_at: 2026-01-01T00:00:00Z
updated_at: 2026-01-01T00:00:00Z
---

## planning
先框定可比的对象集合与统一的比较维度：定位、目标客户、能力、商业模式与近期动态。
为每个对手安排对称任务，避免只对头部玩家深挖而忽略长尾与新进入者。
单列一个任务追踪时效性强的动态，如融资、发布与人事变动。

## task_execution
对每个对手用相同维度采集信息，区分官方口径、媒体报道与第三方分析。
市场规模与份额类数字往往口径不一，记录来源与统计口径而非直接采信。
留意信息时效，竞争态势变化快，优先采用较新的可核对来源。

## reflection
检查是否遗漏了重要竞争者或新兴替代者，各对手是否在同一维度下被比较。
若关键判断依赖单一媒体口径，追加任务寻找独立佐证。

## review
确认对手在一致维度下比较，份额与规模类主张标注了口径且有独立来源。
标记仅凭官方宣传或单一报道得出的判断，提示其可靠性。

## report
用对手对照与格局视角组织，突出差异化定位与变化趋势，而非逐条罗列。
对口径不一或时效敏感的数据明确标注来源与时点。
