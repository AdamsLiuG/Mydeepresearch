---
pack_id: exec-brief
skill_kind: format
schema_version: 1
version: 1
content_hash: "43e9b62d784ebecc"
maturity: verified
status: active
title: 高管一页纸决策简报
topic_domain: general
applicable_when: 当用户要求一页纸总结、高管简报、快速决策摘要或结论先行格式时
match_signals:
  - 一页纸
  - 决策简报
  - 高管摘要
  - executive brief
negative_signals: []
min_match_score: 0.30
stage_scope:
  - report
outline_sections:
  - 核心结论（TL;DR）
  - 关键对比矩阵
  - 决定性依据与数据
  - 风险信号与下一步建议
quality_gates: {}
budget_knobs: {}
layout: {}
derived_from: ""
eval: {}
owner: research-team
created_at: 2026-11-23T00:00:00Z
updated_at: 2026-11-23T00:00:00Z
---

## report
采用「高管决策简报」紧凑交付形态：
1. 开篇直接给出不超过三条的「核心结论（TL;DR）」，每条必须附带合法引用。
2. 中段优先用清晰的 Markdown 对比矩阵表呈现各维度差异，未检索到公开数据的单元格明确标为「未公开/未找到可靠证据」，严禁臆测补全。
3. 末尾用「风险信号（高/中/低）+ 可执行建议」收尾，控制冗余铺垫，强调信息密度与可读性。
