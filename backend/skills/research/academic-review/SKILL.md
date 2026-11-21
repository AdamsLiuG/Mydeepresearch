---
pack_id: academic-review
schema_version: 1
version: 1
content_hash: "62c85a9194fd0d86"
maturity: draft
status: active
supersedes: ""
title: 学术综述
topic_domain: academic_review
applicable_when: 当研究主题涉及学术综述、方法论对比、研究进展梳理或文献追溯时
match_signals:
  - 综述
  - 文献
  - 方法论
  - 研究进展
  - 学术
  - 论文
  - 实验
  - 理论
negative_signals:
  - 政策
  - 合规
  - 竞品
  - 采购
min_match_score: 0.25
stage_scope:
  - planning
  - task_execution
  - reflection
  - review
  - report
quality_gates:
  review_min_sources_per_task: 3
budget_knobs: {}
layout: {}
derived_from: ""
eval: {}
owner: research-team
created_at: 2026-01-01T00:00:00Z
updated_at: 2026-01-01T00:00:00Z
---

## planning
先建立问题谱系：区分主流方法、竞争性方法与已被证伪或过时的方向。
安排任务分别覆盖代表性工作、方法之间的分歧点、以及尚未解决的开放问题。
优先追溯原始工作而非转述，避免综述层层引用带来的失真。

## task_execution
以原始研究为主、综述为辅，记录每个结论所属的方法流派与实验条件。
留意可复现性与适用前提，区分被广泛验证的结论与个别工作的初步主张。
对相互竞争的方法如实并列，不要预设某一派为定论。

## reflection
检查是否覆盖了对立学派与负面结果，是否遗漏了近期的重要进展。
若关键论断仅来自单篇工作，追加任务寻找独立复现或反驳。

## review
确认重要论断可追溯到原始研究，且不同来源不是彼此转述的同一结论。
标记缺乏复现、样本有限或存在方法争议的结论，提示其证据强度。

## report
按问题与方法谱系组织，呈现主流观点、分歧与开放问题，而非罗列文献。
明确区分成熟结论与初步或有争议的结论，标注证据强度。
