from models import (
    ReportClaim,
    ReportConflict,
    ReportEvidenceRef,
    ReportExcerpt,
    ReportScope,
)
from services.red_team_critic import AdversarialRedTeamCritic


def test_confirmed_critic_findings_are_bounded_and_grounded():
    ref = ReportEvidenceRef("T1-S1", None, None, "fixture")
    claims = (
        ReportClaim("conflict", "The rate was 20%.", (ref,), (1,)),
        ReportClaim("population", "The rate for all users was 20%.", (ref,), (1,)),
        ReportClaim("period", "The CY 2025 rate was 20%.", (ref,), (1,)),
        ReportClaim("causal", "The policy caused the rate to fall.", (ref,), (1,)),
        ReportClaim("qualified", "Among active users, the FY 2025 rate was 20%.", (ref,), (1,)),
    )
    excerpts = (ReportExcerpt(ref, "Among active users, the FY 2025 rate was 20%. The policy correlated with the fall."),)
    conflict = ReportConflict("group", ("conflict", "other"), (), "unresolved")
    issues = AdversarialRedTeamCritic.confirm(
        tuple(("sec", claim) for claim in claims), excerpts, (conflict,), (),
    )
    assert {(issue.claim_id, issue.kind) for issue in issues} == {
        ("conflict", "counter_evidence_gap"),
        ("population", "survivorship_bias"),
        ("period", "temporal_skew"),
        ("causal", "causal_overreach"),
    }
    plan = AdversarialRedTeamCritic.plan_revision("digest", issues, round_number=0)
    assert len(plan.issues) == 4
    assert AdversarialRedTeamCritic.plan_revision("digest", issues, round_number=1) is None
    assert claims[-1].scope == ReportScope()


def test_critic_rejects_nine_rows_unknown_targets_and_truncated_answers():
    import json

    from models import ReportModelResponse

    ref = ReportEvidenceRef("T1-S1", None, None, "fixture")
    claim = ReportClaim("fact", "A sample was observed.", (ref,), (1,))
    sample = AdversarialRedTeamCritic.sample((("section", claim),), (ReportExcerpt(ref, claim.text),), (), ())
    row = {
        "section_id": "section", "claim_id": "fact", "kind": "temporal_skew",
        "severity": "high", "proposed_action": "mark_gap", "source_ids": ["T1-S1"],
        "rationale": "Proposed criticism",
    }
    for rows, finish in (([row] * 9, "stop"), ([dict(row, claim_id="unknown")], "stop"), ([], "length")):
        result = AdversarialRedTeamCritic.evaluate(
            ReportModelResponse(json.dumps({"issues": rows}), finish), sample=sample, confirmed=(),
        )
        assert result.status == "invalid"
    assert AdversarialRedTeamCritic.evaluate(None, sample=sample, confirmed=(), reason="timeout").status == "timeout"
    assert AdversarialRedTeamCritic.evaluate(None, sample=sample, confirmed=(), reason="no_model").status == "skipped"
    guess = AdversarialRedTeamCritic.evaluate(
        ReportModelResponse(json.dumps({"issues": [row, row]}), "stop"), sample=sample, confirmed=(),
    )
    assert guess.status == "ok" and guess.issues == ()
