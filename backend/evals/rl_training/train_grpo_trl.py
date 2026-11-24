"""Offline LoRA/QLoRA GRPO policy trainer built on TRL (lazy imports optional training libs)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_EVALS_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _EVALS_ROOT / "src"
for entry in (str(_SRC_ROOT), str(_EVALS_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from evals.rl_training.reward_functions import (  # noqa: E402
    budget_legality_reward_fn,
    json_format_reward_fn,
    verifiable_policy_reward_fn,
)


def load_grpo_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict) and "messages" in parsed:
            rows.append(parsed)
    return rows


def run_dry_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate dataset prompts and score synthetic candidates using the RLVR reward stack."""

    sample_count = len(rows)
    scored_examples = 0
    for row in rows[:8]:
        feats = row.get("state_features") or {}
        cands = row.get("candidate_responses") or []
        if cands:
            _ = verifiable_policy_reward_fn(cands, state_features=[feats for _ in cands])
            scored_examples += 1
    return {
        "dry_run": True,
        "dataset_records": sample_count,
        "scored_groups": scored_examples,
        "status": "ok",
    }


def run_training(
    *,
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    learning_rate: float = 5e-6,
    num_generations: int = 4,
    max_steps: int = 200,
    dry_run: bool = False,
) -> dict[str, Any]:
    rows = load_grpo_dataset(dataset_path)
    if dry_run:
        return run_dry_validation(rows)

    try:  # pragma: no cover - optional heavy GPU dependencies
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - optional heavy GPU dependencies
        raise RuntimeError(
            "TRL/PEFT/datasets not installed in current venv; install offline training extras before GPU training."
        ) from exc

    train_items = [
        {
            "prompt": row["messages"],
            "state_features": row.get("state_features") or {},
        }
        for row in rows
    ]
    dataset = Dataset.from_list(train_items)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_steps=50,
    )
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[
            json_format_reward_fn,
            budget_legality_reward_fn,
            verifiable_policy_reward_fn,
        ],
        args=grpo_cfg,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return {"dry_run": False, "dataset_records": len(rows), "output_dir": str(output_dir), "status": "completed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRL GRPO trainer for DeepResearch routing policy.")
    parser.add_argument("--dataset-path", default="backend/evals/rl_exports/grpo_groups.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="backend/evals/rl_checkpoints/policy_grpo")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="Validate GRPO dataset & RLVR rewards without GPU deps")
    args = parser.parse_args(argv)

    summary = run_training(
        dataset_path=Path(args.dataset_path),
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        dry_run=bool(args.dry_run),
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
