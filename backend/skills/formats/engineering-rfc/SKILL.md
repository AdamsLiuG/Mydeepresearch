---
pack_id: engineering-rfc
skill_kind: format
schema_version: 1
version: 1
content_hash: "b5079770c54bae19"
maturity: verified
status: active
title: 工程架构选型 RFC 文档
topic_domain: tech_selection
applicable_when: 当用户需要架构评审 RFC、技术设计对比文档或工程落地选型方案时
match_signals:
  - RFC
  - 架构评审
  - 技术设计文档
  - 落地迁移方案
negative_signals: []
min_match_score: 0.30
stage_scope:
  - report
outline_sections:
  - 问题背景与工程约束
  - 候选架构对比矩阵
  - 性能、可靠性与成本权衡
  - 推荐方案与渐进迁移路径
  - 回滚预案与运维风险
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
采用「工程架构选型 RFC」交付形态：
1. 严格界定系统边界、吞吐/延迟约束与假设前提，将硬件或测试条件差异显式说明。
2. 构建「候选方案横向矩阵表」，包含架构机制、吞吐与延迟、运维复杂度、生态锁死风险四个固定列，缺证据处标「证据不足」。
3. 在「推荐方案与迁移路径」中区分短期止血改造与中长期演进阶段，明确回滚触发条件。
