"""Judge implementations for benchmark evaluation."""

from evals.judges.heuristic import HeuristicJudge
from evals.judges.llm import LLMJudge

__all__ = ["HeuristicJudge", "LLMJudge"]
