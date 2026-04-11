import sys
import types
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

hello_agents_stub = types.ModuleType("hello_agents")


class DummyToolAwareSimpleAgent:
    responses = []
    calls = []

    def run(self, prompt: str, **kwargs):
        self.last_prompt = prompt
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        if self.responses:
            outcome = self.responses.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return prompt

    def clear_history(self):
        return None


hello_agents_stub.ToolAwareSimpleAgent = DummyToolAwareSimpleAgent
sys.modules.setdefault("hello_agents", hello_agents_stub)

from config import Configuration
from services.planner import PlanningService


class PlannerParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        DummyToolAwareSimpleAgent.responses = []
        DummyToolAwareSimpleAgent.calls = []
        self.service = PlanningService(
            DummyToolAwareSimpleAgent(),
            Configuration.from_env(load_env_file=False),
        )

    def test_extract_tasks_recovers_json_after_tool_calls(self):
        raw_response = """
[TOOL_CALL:note:{"action":"create","task_id":1,"title":"任务 1: 技术架构突破","note_type":"task_state","tags":["deep_research","task_1"],"content":"任务目标：梳理2025年多模态模型在架构设计上的关键突破"}]
[TOOL_CALL:note:{"action":"create","task_id":2,"title":"任务 2: 性能基准评测","note_type":"task_state","tags":["deep_research","task_2"],"content":"任务目标：跟踪主流评测和排名变化"}]
{"tasks":[
  {"title":"技术架构突破","intent":"梳理2025年多模态模型在架构设计上的关键突破","query":"multimodal model 2025 architecture advances"},
  {"title":"性能基准评测","intent":"跟踪主流评测和排名变化","query":"multimodal model 2025 benchmark leaderboard"}
]}
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["title"], "技术架构突破")
        self.assertEqual(tasks[1]["title"], "性能基准评测")

    def test_extract_tasks_recovers_note_tool_calls_without_final_json(self):
        raw_response = """
[TOOL_CALL:note:{"action":"create","task_id":1,"title":"任务 1: 技术架构突破","note_type":"task_state","content":"任务目标：梳理2025年多模态模型在架构设计上的关键突破"}]
[TOOL_CALL:note:{"action":"create","task_id":2,"title":"任务 2: 开源生态动态","note_type":"task_state","content":"任务目标：追踪2025年开源社区发布的多模态模型项目、数据集及工具链更新"}]
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["title"], "技术架构突破")
        self.assertIn("梳理", tasks[0]["intent"])
        self.assertEqual(tasks[1]["title"], "开源生态动态")

    def test_extract_tasks_recovers_note_tool_calls_with_tags_array(self):
        raw_response = """
[TOOL_CALL:note:{"action":"create","task_id":1,"title":"任务 1: 技术架构演进","note_type":"task_state","tags":["deep_research","task_1"],"content":"任务目标：识别核心技术创新与架构突破"}]
[TOOL_CALL:note:{"action":"create","task_id":2,"title":"任务 2: 性能基准对比","note_type":"task_state","tags":["deep_research","task_2"],"content":"任务目标：评估主流模型能力水平与资源消耗"}]
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["title"], "技术架构演进")
        self.assertIn("核心技术创新", tasks[0]["intent"])
        self.assertEqual(tasks[1]["title"], "性能基准对比")

    def test_extract_tasks_recovers_title_intent_and_query_from_tool_content_when_title_is_placeholder(self):
        raw_response = """
[TOOL_CALL:note:{"action":"create","task_id":1,"title":"任务1","note_type":"task_state","content":"任务1: 技术突破调研\\n目标意图: 梳理扩散模型的核心技术突破\\n检索方向: 扩散模型 架构突破 2025"}]
[TOOL_CALL:note:{"action":"create","task_id":2,"title":"任务2","note_type":"task_state","content":"任务2: 应用场景突破\\n目标意图: 识别扩散模型在各垂直领域的商业化落地突破\\n检索方向: 扩散模型应用案例 行业解决方案 2025"}]
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["title"], "技术突破调研")
        self.assertEqual(tasks[0]["intent"], "梳理扩散模型的核心技术突破")
        self.assertEqual(tasks[0]["query"], "扩散模型 架构突破 2025")
        self.assertEqual(tasks[1]["title"], "应用场景突破")
        self.assertEqual(tasks[1]["intent"], "识别扩散模型在各垂直领域的商业化落地突破")
        self.assertEqual(tasks[1]["query"], "扩散模型应用案例 行业解决方案 2025")

    def test_extract_tasks_recovers_key_value_tool_payload_from_content_when_title_missing(self):
        raw_response = """
[TOOL_CALL:note:task_id=2,content="任务2: 应用场景突破\n目标意图: 识别扩散模型在各垂直领域的商业化落地突破\n检索方向: 扩散模型应用案例 行业解决方案 2025"]
[TOOL_CALL:note:task_id=3,content="任务3: 性能效率突破\n目标意图: 分析扩散模型在训练和推理效率上的改进\n检索方向: diffusion model inference acceleration training optimization efficiency 2025"]
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["title"], "应用场景突破")
        self.assertEqual(tasks[0]["intent"], "识别扩散模型在各垂直领域的商业化落地突破")
        self.assertEqual(tasks[0]["query"], "扩散模型应用案例 行业解决方案 2025")
        self.assertEqual(tasks[1]["title"], "性能效率突破")
        self.assertEqual(tasks[1]["intent"], "分析扩散模型在训练和推理效率上的改进")
        self.assertEqual(
            tasks[1]["query"],
            "diffusion model inference acceleration training optimization efficiency 2025",
        )

    def test_extract_tasks_recovers_numbered_text(self):
        raw_response = """
1. 技术架构突破：梳理2025年多模态模型在架构设计上的关键创新
2. 性能基准评测：获取2025年主流多模态模型的权威评测数据与排名变化
3. 应用落地案例：收集多模态模型在医疗、教育、工业等领域的商业化落地实践
4. 开源生态动态：追踪2025年开源社区发布的多模态模型项目、数据集及工具链更新
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 4)
        self.assertEqual(tasks[0]["title"], "技术架构突破")
        self.assertEqual(tasks[3]["title"], "开源生态动态")

    def test_extract_tasks_recovers_markdown_table(self):
        raw_response = """
| 序号 | 任务标题 | 任务目标 | 笔记ID |
| --- | --- | --- | --- |
| 1 | 技术架构演进 | 识别核心技术创新与架构突破 | note_20260323_232540_157 |
| 2 | 性能基准对比 | 评估主流模型能力水平与资源消耗 | note_20260323_232540_158 |
| 3 | 应用场景落地 | 追踪行业商业化应用案例 | note_20260323_232540_159 |
| 4 | 开源生态发展 | 了解社区动态与可访问性 | note_20260323_232540_160 |
| 5 | 挑战与未来方向 | 分析瓶颈及发展趋势预测 | note_20260323_232540_161 |
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 5)
        self.assertEqual(tasks[0]["title"], "技术架构演进")
        self.assertEqual(tasks[0]["intent"], "识别核心技术创新与架构突破")
        self.assertEqual(tasks[4]["title"], "挑战与未来方向")

    def test_extract_tasks_prefers_markdown_table_over_followup_numbered_text(self):
        raw_response = """
| 任务ID | 任务名称 | 核心关注点 | 笔记ID |
| --- | --- | --- | --- |
| 1 | 技术架构演进 | 模型设计原理与训练范式 | note_1 |
| 2 | 主流模型能力对比 | 核心能力与性能差异 | note_2 |
| 3 | 典型应用场景 | 行业落地案例与商业价值 | note_3 |
| 4 | 技术瓶颈与前沿 | 技术局限与未来方向 | note_4 |

1. 交叉验证：检查任务 2 和任务 4 的一致性
2. 综合报告：整合所有任务结论
3. 迭代更新：根据新发现补充说明
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 4)
        self.assertEqual(tasks[0]["title"], "技术架构演进")
        self.assertEqual(tasks[3]["title"], "技术瓶颈与前沿")

    def test_extract_tasks_recovers_task_confirmation_table_without_intent_column(self):
        raw_response = """
| 任务ID | 任务名称 | 笔记ID | 状态 |
|--------|----------|--------|------|
| task_1 | 技术架构与模型创新 | note_1 | 已创建 |
| task_2 | 应用场景与落地实践 | note_2 | 已创建 |
| task_3 | 性能评估与基准测试 | note_3 | 已创建 |
| task_4 | 伦理安全与治理挑战 | note_4 | 已创建 |
| task_5 | 未来趋势与研究方向 | note_5 | 已创建 |

1. 风险前置：先标记潜在争议点
2. 交叉验证：检查任务结论一致性
3. 综合报告：汇总所有发现
4. 迭代更新：补充遗漏细节
"""

        tasks = self.service._extract_tasks(raw_response)

        self.assertEqual(len(tasks), 5)
        self.assertEqual(tasks[0]["title"], "技术架构与模型创新")
        self.assertEqual(tasks[0]["intent"], "聚焦主题的关键问题")
        self.assertEqual(tasks[4]["title"], "未来趋势与研究方向")

    def test_plan_todo_list_repairs_workflow_like_titles(self):
        DummyToolAwareSimpleAgent.responses = [
            """
1. **启动检索**：各Agent可按`query`字段进行网络搜索
2. **进度同步**：每完成一个任务，调用`note`更新对应笔记
3. **交叉验证**：任务2与任务4可相互验证模型性能与发布信息的一致性
4. **综合报告**：所有任务完成后，整合四份笔记输出完整研究报告
""",
            """
{"tasks":[
  {"title":"技术发展脉络","intent":"梳理多模态模型的关键演进路线与阶段性突破","query":"multimodal model development roadmap 2025"},
  {"title":"能力对比","intent":"比较主流多模态模型在理解、生成和推理方面的能力差异","query":"multimodal model benchmark comparison 2025"},
  {"title":"应用场景","intent":"总结医疗、教育、工业等领域的典型落地场景","query":"multimodal model applications 2025"},
  {"title":"瓶颈方向","intent":"分析数据、对齐、推理和部署方面的主要瓶颈与未来趋势","query":"multimodal model bottlenecks future directions 2025"}
]}
""",
        ]

        state = types.SimpleNamespace(research_topic="探索多模态模型在2025年的关键进展")
        todo_items = self.service.plan_todo_list(state)

        self.assertEqual(len(todo_items), 4)
        self.assertEqual(todo_items[0].title, "技术发展脉络")
        self.assertEqual(todo_items[1].title, "能力对比")
        self.assertNotIn("启动检索", [item.title for item in todo_items])

    def test_plan_todo_list_prefers_json_mode_and_falls_back_when_unsupported(self):
        DummyToolAwareSimpleAgent.responses = [
            TypeError("run() got an unexpected keyword argument 'response_format'"),
            '{"tasks":[{"title":"背景梳理","intent":"梳理主题背景","query":"AI agent background"}]}',
        ]

        state = types.SimpleNamespace(research_topic="AI agent")
        todo_items = self.service.plan_todo_list(state)

        self.assertEqual(len(todo_items), 1)
        self.assertEqual(DummyToolAwareSimpleAgent.calls[0]["kwargs"]["response_format"], {"type": "json_object"})
        self.assertEqual(DummyToolAwareSimpleAgent.calls[1]["kwargs"], {})

    def test_plan_todo_list_uses_task_confirmation_table_without_repair(self):
        DummyToolAwareSimpleAgent.responses = [
            """
| 任务ID | 任务名称 | 笔记ID | 状态 |
|--------|----------|--------|------|
| task_1 | 技术架构与融合机制 | note_1 | 已创建 |
| task_2 | 主流模型能力对比 | note_2 | 已创建 |
| task_3 | 典型应用场景与落地 | note_3 | 已创建 |
| task_4 | 评估基准与性能指标 | note_4 | 已创建 |
| task_5 | 技术瓶颈与未来方向 | note_5 | 已创建 |

1. 风险前置：先说明研究边界
2. 交叉验证：校验多个任务结论
3. 综合报告：汇总结论
4. 迭代更新：根据结果补充说明
""",
        ]

        state = types.SimpleNamespace(research_topic="探索多模态大模型的前沿技术")
        todo_items = self.service.plan_todo_list(state)

        self.assertEqual(len(todo_items), 5)
        self.assertEqual(todo_items[0].title, "技术架构与融合机制")
        self.assertEqual(todo_items[4].title, "技术瓶颈与未来方向")

    def test_plan_todo_list_builds_task_specific_query_when_missing(self):
        DummyToolAwareSimpleAgent.responses = [
            """
{"tasks":[
  {"title":"技术架构演进","intent":"梳理关键架构创新"},
  {"title":"应用场景落地","intent":"总结行业应用案例"}
]}
""",
        ]

        state = types.SimpleNamespace(research_topic="探索多模态大模型在2025年的关键进展")
        todo_items = self.service.plan_todo_list(state)

        self.assertEqual(len(todo_items), 2)
        self.assertEqual(
            todo_items[0].query,
            "探索多模态大模型在2025年的关键进展 技术架构演进",
        )
        self.assertEqual(
            todo_items[1].query,
            "探索多模态大模型在2025年的关键进展 应用场景落地",
        )

    def test_plan_todo_list_replaces_raw_query_with_deterministic_canonical_query(self):
        DummyToolAwareSimpleAgent.responses = [
            """
{"tasks":[
  {
    "title":"技术架构演进",
    "intent":"梳理关键架构创新",
    "query":"[TOOL_CALL:note:{\\"note_id\\":\\"note_20260405_001\\"}] 按任务顺序执行 search_web，更新笔记状态，query: 多模态大模型 架构演进 最新"
  }
]}
""",
        ]

        state = types.SimpleNamespace(research_topic="探索多模态大模型在2025年的关键进展")
        todo_items = self.service.plan_todo_list(state)

        self.assertEqual(len(todo_items), 1)
        self.assertEqual(
            todo_items[0].query,
            "探索多模态大模型在2025年的关键进展 技术架构演进",
        )
        self.assertNotIn("note_", todo_items[0].query)
        self.assertNotIn("search_web", todo_items[0].query)

    def test_plan_todo_list_includes_historical_memory_block(self):
        agent = DummyToolAwareSimpleAgent()
        agent.responses = [
            """
{"tasks":[
  {"title":"技术背景","intent":"梳理协议背景","query":"mcp protocol background"}
]}
""",
        ]
        service = PlanningService(
            agent,
            Configuration.from_env(load_env_file=False),
        )

        state = types.SimpleNamespace(research_topic="MCP protocol")
        todo_items = service.plan_todo_list(
            state,
            historical_memory_context="历史研究记忆：过去请求经常遗漏官方文档。",
        )

        self.assertEqual(len(todo_items), 1)
        self.assertIn("HISTORICAL_MEMORY", agent.last_prompt)
        self.assertIn("历史研究记忆", agent.last_prompt)

    def test_plan_todo_list_includes_strategy_memory_block(self):
        agent = DummyToolAwareSimpleAgent()
        agent.responses = [
            """
{"tasks":[
  {"title":"技术背景","intent":"梳理协议背景","query":"mcp protocol background"}
]}
""",
        ]
        service = PlanningService(
            agent,
            Configuration.from_env(load_env_file=False),
        )

        state = types.SimpleNamespace(research_topic="MCP protocol")
        todo_items = service.plan_todo_list(
            state,
            strategy_memory_context="历史策略记忆：优先查官方文档，并规避仅看二手博客的反模式。",
        )

        self.assertEqual(len(todo_items), 1)
        self.assertIn("STRATEGY_MEMORY", agent.last_prompt)
        self.assertIn("历史策略记忆", agent.last_prompt)

    def test_normalize_task_title_prefers_arrow_suffix_and_strips_task_refs(self):
        title = "并行应用层+评估层**（任务2、3）→ 验证技术价值"

        normalized = self.service._normalize_task_title(title)

        self.assertEqual(normalized, "验证技术价值")

    def test_sanitize_tasks_rejects_phase_style_titles_without_real_task_name(self):
        tasks, rejected = self.service._sanitize_tasks(
            [
                {
                    "title": "优先技术层（任务1）",
                    "intent": "先执行第一组任务",
                    "query": "优先技术层 任务1",
                },
                {
                    "title": "技术发展脉络",
                    "intent": "梳理核心技术路线",
                    "query": "多模态模型 技术发展脉络",
                },
            ],
            research_topic="探索多模态大模型在2025年的关键进展",
        )

        self.assertEqual(rejected, 1)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["title"], "技术发展脉络")

    def test_sanitize_tasks_rejects_iteration_update_title(self):
        tasks, rejected = self.service._sanitize_tasks(
            [
                {
                    "title": "迭代更新",
                    "intent": "根据中间结果继续补充工作流说明",
                    "query": "迭代更新",
                },
                {
                    "title": "技术发展脉络",
                    "intent": "梳理核心技术路线",
                    "query": "多模态模型 技术发展脉络",
                },
            ],
            research_topic="探索多模态大模型在2025年的关键进展",
        )

        self.assertEqual(rejected, 1)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["title"], "技术发展脉络")


if __name__ == "__main__":
    unittest.main()
    
    
