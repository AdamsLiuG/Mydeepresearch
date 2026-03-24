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

    def run(self, prompt: str):
        if self.responses:
            return self.responses.pop(0)
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


if __name__ == "__main__":
    unittest.main()
    
    
