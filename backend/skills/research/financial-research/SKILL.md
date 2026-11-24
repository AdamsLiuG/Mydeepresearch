---
pack_id: financial-research
schema_version: 1
version: 1
content_hash: "97b46ca94000f6a3"
maturity: draft
status: active
supersedes: ""
title: 金融研报与财务分析
topic_domain: financial_research
applicable_when: 当研究主题涉及公司财报、财务指标、估值分析、行业研报、营收利润或资本市场披露时
match_signals:
  - 财报
  - 营收
  - 净利润
  - 估值
  - 毛利率
  - 现金流
  - 年报
  - 季报
  - 金融
  - 投研
negative_signals:
  - 学术综述
  - 算法实现
  - 网页抽取
min_match_score: 0.25
stage_scope:
  - planning
  - task_execution
  - reflection
  - review
  - report
quality_gates:
  review_min_sources_per_task: 3
  review_min_domains_per_task: 2
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
    - sec.gov
    - cninfo.com.cn
    - sse.com.cn
    - szse.cn
    - hkexnews.hk
    - reuters.com
    - bloomberg.com
  deprioritized_hosts:
    - jianshu.com
    - csdn.net
    - zhihu.com
    - sina.com
  prior_boost: 0.05
  prior_penalty: -0.03
verifiers:
  - numeric-audit
  - timeline-audit
mcp_tool_scopes:
  allowed_servers:
    - finarag
  allowed_tools:
    - finarag:finarag_list_companies
    - finarag:finarag_filings
    - finarag:finarag_structured_query
    - finarag:finarag_get_report_pages
  auto_query_tools:
    - finarag:finarag_filings
  stage_allowlist:
    - task_execution
    - review
  readonly_only: true
  max_tools_injected: 4
code_sandbox: true
composable_with:
  - competitor-scan
  - policy-analysis
  - tech-selection
derived_from: ""
eval: {}
owner: research-team
created_at: 2026-11-23T00:00:00Z
updated_at: 2026-11-23T00:00:00Z
---

## planning
先锁定披露主体与会计口径：区分合并报表口径与分部口径、GAAP 与 Non-GAAP 口径，不要混比。
拆解任务分别覆盖：官方财报原始披露、同比与环比经营趋势、同业估值/盈利能力对比、以及风险提示。
把「一手交易所/监管公告」与「券商/媒体二手解读」拆成对称验证任务，避免只摘录标题观点。

## task_execution
优先获取交易所公告、年报/季报原文及审计附注，核对货币单位、时间区间和统计口径。
对增长率、毛利率、市占率等派生指标核实分子分母来源，留意一次性损益对利润增速的干扰。
对自媒体股评或未经出处核验的盈利预测保持警惕，必须交叉印证。

## reflection
检查各期财报周期（Q1/H1/FY）是否对齐，是否漏掉重大非经常性损益、监管问询或汇率/政策风险。
若关键财务增速只有二手新闻转述而无公告出处，追加任务回溯原始披露。

## review
核对文中关键金额、同比环比幅度与占比合计是否自洽，确认数据时点与财报期严格匹配。
标注缺少一手公告支撑或口径不一致的财务比较，提醒谨慎采信。

## report
按「核心财务与业务判断—关键指标拆解—同业对比—主要风险与口径提示」组织结构。
所有核心财务数据必须保留具体报告期与会计口径边界，不确定预测明确标出前提假设。
