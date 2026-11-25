"""Budget checks at external transport boundaries."""

from contextlib import contextmanager
from threading import local
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from agent import SafeHelloAgentsLLM
from config import Configuration
from services.evidence import EvidenceStore, FetchPageTool, _ValidatedFetchURL
from services.mcp_bridge import MCPBridgeManager, MCPServerConfig
from services.sandbox_executor import SandboxExecutionRequest, SandboxExecutorService
from services.search import dispatch_search


class DispatchDenied(RuntimeError):
    budget_denied = True


class BudgetStub:
    def __init__(self, limit=10):
        self.limit = limit
        self.attempts = []
        self.stage = "planning"
        self.governor = SimpleNamespace(remaining_timeout=lambda **kwargs: 0.75)

    @contextmanager
    def attempt(self, demand, *, llm=None, required=True):
        if llm and llm[2] is None:
            self.refuse("pricing_unavailable", required=required)
        if len(self.attempts) >= self.limit:
            self.refuse("limit_exceeded", required=required)
        attempt = SimpleNamespace(timeout=0.75, usage=None)
        self.attempts.append((demand, llm, attempt))
        yield attempt

    def refuse(self, reason, *, required=True):
        raise DispatchDenied(reason)


def make_llm(budget, create):
    llm = SafeHelloAgentsLLM.__new__(SafeHelloAgentsLLM)
    llm.model = "demo"
    llm.provider = "custom"
    llm.temperature = 0
    llm.max_tokens = 16
    llm.timeout = 30
    llm.allow_reasoning_fallback = False
    llm._usage_local = local()
    llm.budget_provider = lambda: budget
    llm.budget_pricing_catalog = {
        "custom": {"demo": {
            "prompt_per_1k_tokens": 0.01, "completion_per_1k_tokens": 0.02,
            "cached_prompt_per_1k_tokens": 0.005,
        }},
    }
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm


class BudgetDispatchTests(TestCase):
    def test_llm_compatibility_retry_requires_its_own_permit(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("unknown parameter: stream_options")

        budget = BudgetStub(limit=1)
        llm = make_llm(budget, create)
        with self.assertRaisesRegex(DispatchDenied, "limit_exceeded"):
            llm.invoke([{"role": "user", "content": "hello"}])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["timeout"], 0.75)
        self.assertEqual(calls[0]["max_tokens"], 16)
        self.assertIsNone(budget.attempts[0][2].usage)

    def test_llm_settles_final_usage_with_cache_discount(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        budget = BudgetStub()
        llm = make_llm(budget, lambda **kwargs: [SimpleNamespace(choices=[], usage=usage)])
        llm.invoke([{"role": "user", "content": "hello"}])
        self.assertEqual(budget.attempts[0][2].usage, {
            "input_tokens": 10, "output_tokens": 2, "cost_microusd": 120,
        })

    def test_llm_unknown_pricing_is_refused_before_transport(self):
        create = Mock()
        llm = make_llm(BudgetStub(), create)
        llm.budget_pricing_catalog = {}
        with self.assertRaisesRegex(DispatchDenied, "pricing_unavailable"):
            llm.invoke([{"role": "user", "content": "hello"}])
        create.assert_not_called()

    def test_search_reserves_only_on_cache_miss_and_caps_transport(self):
        budget = BudgetStub()
        config = Configuration(search_api="semanticscholar", search_cache_enabled=True, fetch_full_page=False)
        response = SimpleNamespace(status_code=200, json=lambda: {"data": []})
        with (
            patch("services.search._read_exact_cache", return_value=None),
            patch("services.search._write_cache") as write_cache,
            patch("services.search.requests.get", return_value=response) as transport,
        ):
            dispatch_search("test query", config, 0, budget=budget)
        self.assertEqual(len(budget.attempts), 1)
        self.assertEqual(budget.attempts[0][0], {"search_calls": 1})
        self.assertEqual(transport.call_args.kwargs["timeout"], 0.75)
        self.assertFalse(transport.call_args.kwargs["allow_redirects"])
        entry = write_cache.call_args.args[1]

        with (
            patch("services.search._read_exact_cache", return_value=entry),
            patch("services.search.requests.get") as transport,
        ):
            self.assertTrue(dispatch_search("test query", config, 0, budget=budget)[4])
        transport.assert_not_called()
        self.assertEqual(len(budget.attempts), 1)

    def test_fetch_redirect_requires_a_fresh_permit_before_transport(self):
        budget = BudgetStub(limit=1)
        tool = FetchPageTool(evidence_store=EvidenceStore())
        tool.budget_provider = lambda: budget
        response = MagicMock(status_code=302, headers={"Location": "https://example.org/report"})
        response.__enter__.return_value = response
        with (
            patch("services.evidence._validate_fetch_url", side_effect=lambda url: _ValidatedFetchURL(url, ("8.8.8.8",))),
            patch("services.evidence._PinnedFetchAdapter.send", return_value=response) as transport,
        ):
            with self.assertRaisesRegex(DispatchDenied, "limit_exceeded"):
                tool.run({"task_id": 1, "url": "https://example.org/start"})
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(budget.attempts[0][0], {"web_fetches": 1})
        self.assertEqual(transport.call_args.kwargs["timeout"], 0.75)

    def test_mcp_requires_trusted_tool_resource_bound(self):
        handler = Mock()
        config = Configuration(mcp_enabled=True)
        bridge = MCPBridgeManager(config, servers=[MCPServerConfig(server_id="example", in_process_handler=handler)])
        try:
            with self.assertRaisesRegex(DispatchDenied, "dispatch_unbounded"):
                bridge.call_tool(
                    server_id="example", tool_name="tool", budget=BudgetStub(),
                    active_scopes={"allowed_servers": ["example"], "allowed_tools": ["example:tool"]},
                )
        finally:
            bridge.close()
        handler.assert_not_called()

    def test_sandbox_denial_stops_backend_after_local_validation(self):
        backend = Mock(backend_name="stub")
        service = SandboxExecutorService(enabled=True, backend_preference="docker", docker_backend=backend)
        service.budget_provider = lambda: BudgetStub(limit=0)
        request = SandboxExecutionRequest(task_id=1, code="print(1)", derived_from_source_ids=["T1-S1"])
        with self.assertRaisesRegex(DispatchDenied, "limit_exceeded"):
            service.execute(request, valid_source_ids={"T1-S1"})
        backend.execute.assert_not_called()

    def test_opaque_search_adapter_does_not_call_sdk(self):
        config = Configuration(search_api="duckduckgo", search_cache_enabled=False)
        with patch("services.search._GLOBAL_SEARCH_TOOL.run") as transport:
            with self.assertRaisesRegex(DispatchDenied, "dispatch_unbounded"):
                dispatch_search("test query", config, 0, budget=BudgetStub())
        transport.assert_not_called()
