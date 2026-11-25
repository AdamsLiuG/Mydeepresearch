"""End-to-end verification tests for DeepResearch v4-06 closing-loop capabilities."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"
for entry in (str(BACKEND_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from config import Configuration  # noqa: E402
from evals.compare_variants import (  # noqa: E402
    build_comparison,
    paired_bootstrap_ci,
    render_markdown,
)
from evals.export_rl_trajectories import (  # noqa: E402
    export_grpo_multiturn_records,
)
from evals.export_rl_trajectories import (
    main as export_rl_main,
)
from evals.loader import load_benchmark_cases  # noqa: E402
from evals.report_metrics import summarize_paired_eval_comparison  # noqa: E402
from evals.rl_training.reward_functions import (
    multiturn_trajectory_reward_fn,  # noqa: E402
)
from evals.rl_training.train_grpo_trl import (  # noqa: E402
    main as train_grpo_main,
)
from evals.rl_training.train_grpo_trl import (
    run_dry_validation,
)
from main import create_app  # noqa: E402
from services.finarag_mcp_server import (  # noqa: E402
    FinaRAGAdapter,
    FinaragFilingsInput,
    FinaRAGMCPServer,
    TenantACL,
)
from services.policy_controller import PolicyStateFeatures  # noqa: E402
from services.sandbox_executor import (  # noqa: E402
    HttpSandboxBackend,
    SandboxExecutionRequest,
    SandboxExecutorService,
    _validate_sidecar_url,
)
from services.sandbox_sidecar import create_sidecar_app  # noqa: E402
from services.storage_backends import build_request_state_store  # noqa: E402


def _make_multiturn_rollout(request_id: str, rollout_id: str, *, reward_tier: str) -> dict[str, Any]:
    s1 = PolicyStateFeatures(
        task_id=1, round_index=1, max_rounds=3, evidence_count=1,
        distinct_domains=1, high_quality_sources=0, has_gap_signals=True,
        searches_left=2, fetches_left=1, remaining_budget_ratio=0.8,
        task_title="Verify Q4 filing", task_intent="Gather primary evidence",
        gap_signals=("low_source_diversity",), can_fetch=False,
    )
    s2 = PolicyStateFeatures(
        task_id=1, round_index=2, max_rounds=3, evidence_count=4,
        distinct_domains=3, high_quality_sources=2, has_gap_signals=False,
        searches_left=1, fetches_left=1, remaining_budget_ratio=0.6,
        task_title="Verify Q4 filing", task_intent="Stop once verified",
        gap_signals=(), can_fetch=True,
    )
    if reward_tier == "high":
        support = "supported"
        clean_issues = 0
        faithfulness = 0.94
        hybrid = 0.90
        verifier_count = 0
        tokens = 3500
    else:
        support = "unverified_claim"
        clean_issues = 2
        faithfulness = 0.45
        hybrid = 0.40
        verifier_count = 2
        tokens = 18500
    return {
        "request_id": request_id,
        "rollout_id": rollout_id,
        "case_id": "case-multiturn-1",
        "group_id": "grp-mt-1",
        "topic": "FinaRAG multi-turn policy verification",
        "status": "success",
        "skill_pack": {"active_verifiers": ["numeric-audit"]},
        "todo_items": [
            {"id": 1, "status": "completed", "claims": [{"text": "Margin +4%", "support_status": support}]},
        ],
        "request_metrics": {
            "total_tasks": 1,
            "completed_tasks": 1,
            "total_tokens": tokens,
            "elapsed_ms": 9000.0,
            "judge_metrics": {
                "reference_match_rate": faithfulness,
                "grounded_bullet_ratio": faithfulness,
                "hybrid_score": hybrid,
            },
            "review_summary": {
                "critical_issues": clean_issues,
                "warning_issues": 0,
                "verifiers_executed": ["numeric-audit"],
                "verifier_issue_count": verifier_count,
            },
            "policy_decisions": [
                {
                    "applied": True,
                    "features": s1.to_dict(),
                    "decision": {
                        "action": "search",
                        "confidence": 0.85,
                        "reason": "need diversified sources",
                        "source": "model",
                    },
                },
                {
                    "applied": True,
                    "features": s2.to_dict(),
                    "decision": {
                        "action": "stop",
                        "confidence": 0.92,
                        "reason": "evidence sufficient",
                        "source": "model",
                    },
                },
            ],
        },
    }


class TestV4ClosingLoop(unittest.TestCase):
    def test_golden_32_benchmark_preserves_24_and_covers_new_domains(self) -> None:
        bench24 = load_benchmark_cases(BACKEND_DIR / "evals" / "benchmarks" / "golden_multidomain_24cases.jsonl")
        bench32 = load_benchmark_cases(BACKEND_DIR / "evals" / "benchmarks" / "golden_multidomain_32cases.jsonl")

        self.assertEqual(len(bench24), 24)
        self.assertEqual(len(bench32), 32)
        self.assertEqual(len({case.id for case in bench32}), 32)
        for old_case, new_case in zip(bench24, bench32[:24]):
            self.assertEqual(old_case, new_case)

        new_domains = {case.metadata.get("domain") for case in bench32[24:]}
        self.assertEqual(
            new_domains,
            {"finance_filing", "legal_compliance", "clinical_biomed", "temporal_evolution", "red_team_conflict"},
        )
        new_languages = {case.metadata.get("language") for case in bench32[24:]}
        self.assertEqual(new_languages, {"zh", "en"})

    def test_paired_bootstrap_ci_and_comparison_stats(self) -> None:
        empty_ci = paired_bootstrap_ci([])
        self.assertEqual(empty_ci["n"], 0)
        self.assertFalse(empty_ci["significant"])

        const_ci = paired_bootstrap_ci([0.25, 0.25, 0.25], seed=42)
        self.assertEqual(const_ci["ci_low"], 0.25)
        self.assertEqual(const_ci["ci_high"], 0.25)
        self.assertTrue(const_ci["significant"])

        ci_a = paired_bootstrap_ci([0.1, 0.2, 0.15, 0.3, 0.05], n_bootstrap=200, seed=7)
        ci_b = paired_bootstrap_ci([0.1, 0.2, 0.15, 0.3, 0.05], n_bootstrap=200, seed=7)
        self.assertEqual(ci_a, ci_b)
        self.assertTrue(ci_a["significant"])

        baseline = {
            "variant": "v3",
            "results": [
                {
                    "id": "c1",
                    "topic": "T1",
                    "metrics": {
                        "reference_match_rate": 0.6,
                        "grounded_bullet_ratio": 0.6,
                        "estimated_cost": 0.04,
                        "total_latency_ms": 1200.0,
                    },
                },
                {
                    "id": "c2",
                    "topic": "T2",
                    "metrics": {
                        "reference_match_rate": 0.5,
                        "grounded_bullet_ratio": 0.5,
                        "estimated_cost": 0.05,
                        "total_latency_ms": 1500.0,
                    },
                },
            ],
        }
        candidate = {
            "variant": "v4",
            "results": [
                {
                    "id": "c1",
                    "topic": "T1",
                    "metrics": {
                        "reference_match_rate": 0.85,
                        "grounded_bullet_ratio": 0.85,
                        "estimated_cost": 0.02,
                        "total_latency_ms": 900.0,
                    },
                },
                {
                    "id": "c2",
                    "topic": "T2",
                    "metrics": {
                        "reference_match_rate": 0.75,
                        "grounded_bullet_ratio": 0.75,
                        "estimated_cost": 0.03,
                        "total_latency_ms": 1000.0,
                    },
                },
            ],
        }
        comparison = build_comparison(baseline, candidate, bootstrap_samples=200, bootstrap_seed=42)
        self.assertIn("paired_stats", comparison)
        cost_stat = comparison["paired_stats"]["estimated_cost"]
        self.assertEqual(cost_stat["direction"], "lower_is_better")
        self.assertEqual(cost_stat["wins"], 2)
        self.assertLess(cost_stat["paired_delta"], 0.0)

        md = render_markdown(comparison)
        self.assertIn("Paired Bootstrap Confidence Intervals", md)

        report_summary = summarize_paired_eval_comparison(baseline, candidate, n_bootstrap=200, seed=42)
        self.assertEqual(report_summary["verdict"], "pass")
        self.assertGreater(report_summary["bootstrap_summary"]["ci_low"], 0.0)

    def test_multiturn_grpo_export_generic_mix_and_cpu_dry_run(self) -> None:
        rollouts = [
            _make_multiturn_rollout("req-h1", "r-h1", reward_tier="high"),
            _make_multiturn_rollout("req-h2", "r-h2", reward_tier="high"),
            _make_multiturn_rollout("req-h3", "r-h3", reward_tier="high"),
            _make_multiturn_rollout("req-l1", "r-l1", reward_tier="low"),
            _make_multiturn_rollout("req-l2", "r-l2", reward_tier="low"),
            _make_multiturn_rollout("req-l3", "r-l3", reward_tier="low"),
        ]
        exported = export_grpo_multiturn_records(rollouts, generic_mix_ratio=0.15, seed=42)
        domain_rows = [row for row in exported if row.get("sample_source") == "domain_multiturn"]
        generic_rows = [row for row in exported if row.get("sample_source") == "generic_replay"]
        self.assertEqual(len(domain_rows), 6)
        self.assertEqual(len(generic_rows), 1)
        self.assertTrue(all(len(row["turns"]) == 2 for row in domain_rows))
        self.assertNotEqual(domain_rows[0]["turns"][0]["discounted_return"], 0.0)

        validation = run_dry_validation(exported, mode="grpo-multiturn", generic_mix_ratio=0.15)
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["domain_records"], 6)
        self.assertEqual(validation["generic_records"], 1)
        self.assertAlmostEqual(validation["actual_generic_mix_ratio"], round(1 / 7, 4))

        good_completion = domain_rows[0]["reference_completion"]
        rewards = multiturn_trajectory_reward_fn(
            [good_completion, "not-json"],
            state_features=[domain_rows[0]["state_features"], domain_rows[0]["state_features"]],
            verifiable_oracle=[domain_rows[0]["verifiable_oracle"], domain_rows[0]["verifiable_oracle"]],
            reference_completion=[good_completion, good_completion],
            step_advantage=[domain_rows[0]["step_advantage"], domain_rows[0]["step_advantage"]],
            turns=[domain_rows[0]["turns"], domain_rows[0]["turns"]],
        )
        self.assertGreater(rewards[0], 0.3)
        self.assertEqual(rewards[1], -1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "snaps"
            out_dir = Path(tmpdir) / "out"
            in_dir.mkdir()
            for item in rollouts:
                (in_dir / f"{item['request_id']}.json").write_text(json.dumps(item), encoding="utf-8")
            rc_export = export_rl_main([
                "--input-dir", str(in_dir),
                "--output-dir", str(out_dir),
                "--mode", "grpo-multiturn",
                "--generic-mix-ratio", "0.15",
            ])
            self.assertEqual(rc_export, 0)
            mt_path = out_dir / "grpo_multiturn_records.jsonl"
            self.assertTrue(mt_path.exists())
            rc_train = train_grpo_main([
                "--dataset-path", str(mt_path),
                "--dry-run",
                "--mode", "grpo-multiturn",
                "--generic-mix-ratio", "0.15",
            ])
            self.assertEqual(rc_train, 0)

    def test_finarag_hybrid_dense_bm25_rrf_and_fail_closed_acl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports = root / "databases" / "chunked_reports"
            reports.mkdir(parents=True)
            (root / "subset.csv").write_text(
                "company_name,sha1\nAcme Corp,sha_acme\nSecret Corp,sha_secret\n",
                encoding="utf-8",
            )
            doc = {
                "metainfo": {"company_name": "Acme Corp", "sha1_name": "sha_acme"},
                "content": {
                    "pages": [
                        {"page": 1, "text": "Unrelated corporate boilerplate appendix note."},
                        {"page": 2, "text": "EBITDA margin overview without strategic drivers."},
                        {"page": 3, "text": "Recurring gross earnings and free surplus yield expanded."},
                        {"page": 4, "text": "EBITDA expansion driven by recurring operating profitability cash generation."},
                    ]
                },
            }
            (reports / "acme.json").write_text(json.dumps(doc), encoding="utf-8")
            (reports / "secret.json").write_text(
                json.dumps({**doc, "metainfo": {"company_name": "Secret Corp", "sha1_name": "sha_secret"}}),
                encoding="utf-8",
            )

            def _semantic_embedder(text: str) -> list[float]:
                lowered = text.lower()
                if (
                    "operating profitability" in lowered
                    or "cash generation" in lowered
                    or "gross earnings" in lowered
                ):
                    return [1.0, 0.2, 0.0]
                if "ebitda" in lowered:
                    return [0.3, 0.9, 0.0]
                return [0.0, 0.0, 1.0]

            acl = TenantACL({"team_a": ["Acme Corp"]})
            legacy = FinaRAGAdapter(data_root=root, acl=acl, hybrid_rrf_enabled=False, embedder=_semantic_embedder)
            legacy_out = legacy.retrieve_filings(
                FinaragFilingsInput(
                    company_name="Acme Corp",
                    query="operating profitability cash generation",
                    retrieval_mode="hybrid",
                    top_n=4,
                ),
                tenant_id="team_a",
            )
            self.assertEqual(legacy_out.retrieval_mode, "bm25")
            self.assertTrue(legacy_out.notices)
            # Without hybrid RRF, semantic paraphrase on page 3 (zero lexical overlap) is not recalled
            self.assertEqual([p.page_index_1based for p in legacy_out.passages], [4])

            rrf_adapter = FinaRAGAdapter(
                data_root=root,
                acl=acl,
                hybrid_rrf_enabled=True,
                rrf_k=60,
                embedder=_semantic_embedder,
            )
            rrf_out = rrf_adapter.retrieve_filings(
                FinaragFilingsInput(
                    company_name="Acme Corp",
                    query="operating profitability cash generation",
                    retrieval_mode="hybrid",
                    top_n=4,
                ),
                tenant_id="team_a",
            )
            self.assertEqual(rrf_out.retrieval_mode, "hybrid_rrf")
            self.assertEqual(rrf_out.notices, [])
            self.assertEqual(rrf_out.passages[0].page_index_1based, 4)
            self.assertIsNotNone(rrf_out.passages[0].bm25_rank)
            self.assertIsNotNone(rrf_out.passages[0].dense_rank)
            expected_top_rrf = round(1.0 / (60 + rrf_out.passages[0].bm25_rank) + 1.0 / (60 + rrf_out.passages[0].dense_rank), 6)
            self.assertAlmostEqual(rrf_out.passages[0].rrf_score, expected_top_rrf)

            # Tenant ACL remains strictly fail-closed under hybrid RRF
            with self.assertRaises(PermissionError):
                rrf_adapter.retrieve_filings(
                    FinaragFilingsInput(company_name="Secret Corp", query="EBITDA", retrieval_mode="hybrid"),
                    tenant_id="team_a",
                )
            server = FinaRAGMCPServer(rrf_adapter)
            self.assertIsNotNone(server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "ping"}))

    def test_sandbox_http_sidecar_auth_ast_and_executor_integration(self) -> None:
        unconfigured = TestClient(create_sidecar_app(auth_secret="", allow_unauthenticated=False))
        self.assertEqual(unconfigured.get("/v1/capabilities").status_code, 503)

        sidecar_app = create_sidecar_app(auth_secret="sidecar-secret-123")
        client = TestClient(sidecar_app)
        self.assertEqual(client.get("/v1/capabilities").status_code, 401)
        self.assertEqual(
            client.post(
                "/v1/executions",
                headers={"Authorization": "Bearer wrong-token"},
                json={"code": "result = 1 + 1"},
            ).status_code,
            401,
        )

        auth_headers = {
            "Authorization": "Bearer sidecar-secret-123",
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        }
        caps = client.get("/v1/capabilities", headers=auth_headers)
        self.assertEqual(caps.status_code, 200)
        self.assertEqual(caps.json()["service"], "sandbox-sidecar")
        self.assertEqual(caps.headers.get("traceparent"), auth_headers["traceparent"])

        ast_blocked = client.post(
            "/v1/executions",
            headers=auth_headers,
            json={"code": "import os\nresult = os.getcwd()"},
        )
        self.assertEqual(ast_blocked.status_code, 200)
        self.assertFalse(ast_blocked.json()["ok"])
        self.assertEqual(ast_blocked.json()["error_code"], "ast_rejected")

        # Validate SSRF/scheme guard
        with self.assertRaises(ValueError):
            _validate_sidecar_url("file:///etc/passwd")
        with self.assertRaises(ValueError):
            _validate_sidecar_url("http://user:pass@127.0.0.1:8090")

        # Execute end-to-end through SandboxExecutorService via sync MockTransport -> FastAPI TestClient
        def _sidecar_dispatch(req: httpx.Request) -> httpx.Response:
            replied = client.request(
                method=req.method,
                url=req.url.raw_path.decode("ascii"),
                headers=dict(req.headers),
                content=req.content,
            )
            return httpx.Response(
                status_code=replied.status_code,
                headers=dict(replied.headers),
                content=replied.content,
            )

        transport = httpx.MockTransport(_sidecar_dispatch)
        backend = HttpSandboxBackend(
            base_url="http://sidecar.local",
            auth_secret="sidecar-secret-123",
            timeout_seconds=10.0,
            transport=transport,
        )
        executor = SandboxExecutorService(
            enabled=True,
            backend_preference="sidecar",
            sidecar_backend=backend,
        )
        result = executor.execute(
            SandboxExecutionRequest(
                task_id=1,
                code="total = sum(row['val'] for row in rev_table)\nresult = {'total': total}",
                tables=[
                    {
                        "name": "rev_table",
                        "source_id": "T1-S1",
                        "columns": ["val"],
                        "rows": [[10], [25]],
                    }
                ],
                derived_from_source_ids=["T1-S1"],
                traceparent=auth_headers["traceparent"],
            ),
            valid_source_ids={"T1-S1"},
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.backend, "sidecar")
        self.assertEqual(result.result, {"total": 35})

        disabled_cfg = Configuration.from_env(
            overrides={
                "sandbox_enabled": True,
                "active_code_sandbox_enabled": True,
                "sandbox_backend": "sidecar",
                "v4_sandbox_sidecar_enabled": False,
            },
            load_env_file=False,
        )
        self.assertFalse(disabled_cfg.resolved_sandbox_effective_enabled())
        enabled_cfg = Configuration.from_env(
            overrides={
                "sandbox_enabled": True,
                "active_code_sandbox_enabled": True,
                "sandbox_backend": "sidecar",
                "v4_sandbox_sidecar_enabled": True,
            },
            load_env_file=False,
        )
        self.assertTrue(enabled_cfg.resolved_sandbox_effective_enabled())

    def test_workbench_annotations_reverify_and_revision_diff_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = Configuration.from_env(
                overrides={
                    "request_state_enabled": True,
                    "request_state_dir": tmpdir,
                    "tenant_isolation_enabled": True,
                    "auth_enabled": True,
                    "tenant_api_keys": {
                        "key-alpha-secret": "tenant-alpha",
                        "key-beta-secret": "tenant-beta",
                    },
                    "v4_workbench_annotations_enabled": False,
                },
                load_env_file=False,
            )
            store = build_request_state_store(base_cfg)
            assert store is not None
            seed_snapshot = {
                "request_id": "req-ann-1",
                "tenant_id": "tenant-alpha",
                "topic": "FinaRAG 季度营收对比",
                "status": "success",
                "phase": "completed",
                "report_markdown": "## 核心结论\n\n营收同比增速稳定 [T1-S1]。\n\n## 风险\n\n需关注汇兑波动。\n",
                "report_workbench": {
                    "citation_index": {
                        "T1-S1": {
                            "source_id": "T1-S1",
                            "title": "FY2025 Q4 Filing",
                            "snippet": "核心结论：Q4 营收同比增速保持 18.4%",
                            "url": "https://sec.example.com/filing",
                        }
                    },
                    "evidence_conflicts": [],
                },
                "todo_items": [],
            }
            store.save("req-ann-1", seed_snapshot, tenant_id="tenant-alpha")

            # Disabled flag returns 404 on every annotation/reverify/revision route
            disabled_app = create_app(base_config=base_cfg)
            disabled_client = TestClient(disabled_app)
            alpha_headers = {"X-API-Key": "key-alpha-secret"}
            self.assertEqual(disabled_client.get("/requests/req-ann-1/annotations", headers=alpha_headers).status_code, 404)
            self.assertEqual(
                disabled_client.post(
                    "/requests/req-ann-1/annotations",
                    headers=alpha_headers,
                    json={"section_id": "核心结论", "comment": "check"},
                ).status_code,
                404,
            )

            enabled_cfg = base_cfg.model_copy(update={"v4_workbench_annotations_enabled": True})
            app = create_app(base_config=enabled_cfg)
            client = TestClient(app)

            # Unauthenticated fails closed with 401; cross-tenant isolation returns 404 (not 403/200)
            self.assertEqual(client.get("/requests/req-ann-1/annotations").status_code, 401)
            beta_headers = {"X-API-Key": "key-beta-secret"}
            self.assertEqual(client.get("/requests/req-ann-1/annotations", headers=beta_headers).status_code, 404)
            self.assertEqual(
                client.post(
                    "/requests/req-ann-1/reverify",
                    headers=beta_headers,
                    json={"section_id": "核心结论"},
                ).status_code,
                404,
            )

            # Validation 422 for blank comment / empty reverify target
            self.assertEqual(
                client.post(
                    "/requests/req-ann-1/annotations",
                    headers=alpha_headers,
                    json={"section_id": "核心结论", "comment": "   "},
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    "/requests/req-ann-1/reverify",
                    headers=alpha_headers,
                    json={"section_id": ""},
                ).status_code,
                422,
            )

            # Append annotation with valid CAS state_version
            initial_get = client.get("/requests/req-ann-1/annotations", headers=alpha_headers)
            self.assertEqual(initial_get.status_code, 200)
            v0 = initial_get.json()["state_version"]
            created = client.post(
                "/requests/req-ann-1/annotations",
                headers=alpha_headers,
                json={
                    "section_id": "核心结论",
                    "comment": "请复核 Q4 营收增速口径 [T1-S1]",
                    "severity": "concern",
                    "expected_state_version": v0,
                },
            )
            self.assertEqual(created.status_code, 200)
            ann_id = created.json()["annotation"]["annotation_id"]
            v1 = created.json()["state_version"]
            self.assertGreater(v1, v0)

            # Stale expected_state_version triggers 409 CAS conflict
            stale = client.post(
                "/requests/req-ann-1/annotations",
                headers=alpha_headers,
                json={"section_id": "风险", "comment": "stale", "expected_state_version": v0},
            )
            self.assertEqual(stale.status_code, 409)

            # Targeted reverify creates immutable revision 2 while revision 1 stays untouched
            reverified = client.post(
                "/requests/req-ann-1/reverify",
                headers=alpha_headers,
                json={
                    "annotation_id": ann_id,
                    "section_id": "核心结论",
                    "expected_state_version": v1,
                },
            )
            self.assertEqual(reverified.status_code, 200)
            rev_data = reverified.json()
            self.assertEqual(rev_data["summary"]["status"], "supported")
            self.assertIn("T1-S1", rev_data["summary"]["matched_source_ids"])
            self.assertEqual(rev_data["revision"]["revision_id"], 2)

            revisions_resp = client.get("/requests/req-ann-1/revisions", headers=alpha_headers)
            self.assertEqual(revisions_resp.status_code, 200)
            revs = revisions_resp.json()["revisions"]
            self.assertEqual(len(revs), 2)
            self.assertEqual(revs[0]["report_markdown"], seed_snapshot["report_markdown"])
            self.assertNotEqual(revs[0]["content_sha256"], revs[1]["content_sha256"])

            # Diff between revision 1 and 2
            diff_resp = client.get(
                "/requests/req-ann-1/revisions/diff?base=1&target=2",
                headers=alpha_headers,
            )
            self.assertEqual(diff_resp.status_code, 200)
            diff = diff_resp.json()
            self.assertGreater(diff["added_count"], 0)
            self.assertIn("核心结论", diff["changed_sections"])
            self.assertEqual(
                client.get("/requests/req-ann-1/revisions/diff?base=1&target=99", headers=alpha_headers).status_code,
                404,
            )
