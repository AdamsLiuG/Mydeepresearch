"""TRL LoRA training and veRL dataset conversion, with CPU-only replay validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for entry in (_BACKEND_DIR, _BACKEND_DIR / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evals.export_rl_trajectories import features_from_record  # noqa: E402
from evals.rl_training.reward_functions import verifiable_policy_reward_fn  # noqa: E402
from services.policy_controller import (  # noqa: E402
    format_policy_messages,
    is_budget_legal_action,
    parse_policy_decision,
)

_EXPORT_DIR = _BACKEND_DIR / "evals" / "rl_exports"
_DEFAULT_OUTPUT = _BACKEND_DIR / "evals" / "rl_checkpoints" / "policy_grpo"
_ORACLE_RATES = ("final_citation_faithfulness", "final_hybrid_score", "final_verifier_clean_rate")


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and abs(value) <= sys.float_info.max


def _validated_item(row: dict[str, Any]) -> dict[str, Any]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError("expected a two-message policy prompt")
    oracle = row.get("verifiable_oracle")
    if not isinstance(oracle, dict):
        raise ValueError("expected a verifiable oracle object")
    # Older per-step blueprint rows carry most features in the serialized user state.
    user = prompt[1]
    if not isinstance(user, dict) or not isinstance(user.get("content"), str):
        raise ValueError("expected a serialized user state")
    payload = json.loads(user["content"])
    if not isinstance(payload, dict):
        raise ValueError("expected a structured policy input")
    features = features_from_record(row.get("state_features", payload.get("state", payload)))
    if features is None or prompt != format_policy_messages(
        features, prompt_version=payload.get("prompt_version", "policy_prompt_v2"),
    ):
        raise ValueError("prompt and state features do not match the serving policy prompt")
    state = features.to_dict()
    if any(key in oracle and oracle[key] != state[key] for key in state):
        raise ValueError("oracle state and policy prompt disagree")
    for key in _ORACLE_RATES:
        if not _finite_number(oracle.get(key)) or not 0 <= oracle[key] <= 1:
            raise ValueError(f"expected a finite oracle rate in [0, 1]: {key}")
    high = oracle.get("high_severity_issues_after_step", 0)
    if type(high) is not int or high < 0:
        raise ValueError("expected a nonnegative integer severity count")
    reference = row.get("reference_completion")
    decision = parse_policy_decision(reference) if isinstance(reference, str) else None
    if decision is None or not is_budget_legal_action(features, decision):
        raise ValueError("expected a schema- and budget-legal reference completion")
    advantage = row.get("step_advantage", row.get("group_advantage"))
    if not _finite_number(advantage):
        raise ValueError("expected a finite replay advantage")
    return {
        "prompt": prompt,
        "state_features": state,
        "reference_completion": reference,
        "step_advantage": advantage,
        "verifiable_oracle": {
            **{key: oracle[key] for key in _ORACLE_RATES},
            "high_severity_issues_after_step": high,
        },
    }


def load_grpo_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                if isinstance(row.get("messages"), list) and isinstance(row.get("candidate_responses"), list):
                    if not row["candidate_responses"] or features_from_record(row.get("state_features")) is None:
                        raise ValueError("invalid legacy group")
                else:
                    _validated_item(row)
            except (ValueError, TypeError, RecursionError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"No GRPO training rows found in {path}")
    return rows


def build_grpo_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve prompt, decision reference and trusted oracle as TRL reward columns."""

    if not rows:
        raise ValueError("No GRPO training rows")
    if any("candidate_responses" in row for row in rows):
        raise ValueError("Training requires grpo_step_records.jsonl with a per-step oracle")
    return [_validated_item(row) for row in rows]


def verl_dataset_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build veRL's Parquet schema with reward fields in extra_info."""

    return [
        {
            "prompt": item["prompt"],
            "data_source": "deepresearch_policy",
            "reward_model": {"style": "rule", "ground_truth": item["reference_completion"]},
            "extra_info": {"index": index, **{key: value for key, value in item.items() if key != "prompt"}},
        }
        for index, item in enumerate(build_grpo_dataset(rows))
    ]


def run_dry_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No GRPO training rows")
    for row in rows:
        if "candidate_responses" in row:
            completions = row["candidate_responses"]
            features = row.get("state_features")
            scores = verifiable_policy_reward_fn(completions, state_features=[features] * len(completions))
        else:
            item = _validated_item(row)
            kwargs = {key: [value] for key, value in item.items() if key != "prompt"}
            scores = verifiable_policy_reward_fn([item["reference_completion"]], **kwargs)
        if not scores or any(not math.isfinite(score) for score in scores):
            raise ValueError("Replay reward validation failed")
    return {"dry_run": True, "dataset_records": len(rows), "scored_groups": len(rows), "status": "ok"}


def run_training(
    *,
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    learning_rate: float = 5e-6,
    num_generations: int = 8,
    max_steps: int = 200,
    dry_run: bool = False,
    warm_start_model: str | None = None,
    sft_warmup: bool = False,
    use_vllm: bool = False,
) -> dict[str, Any]:
    rows = load_grpo_dataset(dataset_path)
    if dry_run:
        return run_dry_validation(rows)

    if not _finite_number(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if type(num_generations) is not int or not 4 <= num_generations <= 8:
        raise ValueError("num_generations must be between 4 and 8")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    if warm_start_model and sft_warmup:
        raise ValueError("Choose either an SFT model checkpoint or an SFT warmup")

    items = build_grpo_dataset(rows)
    try:  # pragma: no cover - optional GPU dependencies
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("GPU training requires a TRL environment with PEFT and datasets available.") from exc

    tokenizer = AutoTokenizer.from_pretrained(warm_start_model or model_name, trust_remote_code=False)
    # TRL 0.21 does not pass custom template kwargs during GRPO generation.
    if isinstance(tokenizer.chat_template, str) and "enable_thinking" in tokenizer.chat_template:
        tokenizer.chat_template = "{% set enable_thinking = false %}" + tokenizer.chat_template

    warmup_count = 0
    if sft_warmup:
        try:  # pragma: no cover - optional GPU dependencies
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("SFT warmup requires SFTTrainer in the TRL environment.") from exc
        warmup_items = [
            {"prompt": item["prompt"], "completion": [{"role": "assistant", "content": item["reference_completion"]}]}
            for row, item in zip(rows, items)
            if _finite_number(row.get("trajectory_reward")) and row["trajectory_reward"] >= 0.70
        ]
        if not warmup_items:
            raise ValueError("SFT warmup requires per-step rows with trajectory reward of at least 0.70")
        warmup_count = len(warmup_items)
        warmup_dir = output_dir / "sft_warmup"
        sft_trainer = SFTTrainer(
            model=model_name, processing_class=tokenizer,
            args=SFTConfig(
                output_dir=str(warmup_dir), num_train_epochs=1, bf16=True, max_length=1792,
                per_device_train_batch_size=1, completion_only_loss=True,
            ),
            train_dataset=Dataset.from_list(warmup_items),
        )
        sft_trainer.train()
        sft_trainer.save_model(str(warmup_dir))
        del sft_trainer
        warm_start_model = str(warmup_dir)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM",
    )
    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir), learning_rate=learning_rate, num_generations=num_generations,
        max_steps=max_steps, per_device_train_batch_size=4, gradient_accumulation_steps=8,
        steps_per_generation=num_generations, max_prompt_length=1536, max_completion_length=256,
        beta=0.04, epsilon=0.2, loss_type="dr_grpo", scale_rewards=True,
        temperature=0.8, top_p=0.95, lr_scheduler_type="cosine", warmup_ratio=0.05,
        bf16=True, optim="adamw_torch_fused", remove_unused_columns=False,
        use_vllm=use_vllm, vllm_mode="colocate", logging_steps=5, save_steps=50,
    )
    trainer = GRPOTrainer(
        model=warm_start_model or model_name, reward_funcs=verifiable_policy_reward_fn,
        args=grpo_cfg, train_dataset=Dataset.from_list(items), peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return {
        "dry_run": False, "dataset_records": len(rows), "sft_warmup_records": warmup_count,
        "output_dir": str(output_dir), "status": "completed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRL GRPO trainer for DeepResearch routing policy.")
    parser.add_argument("--dataset-path", type=Path, default=_EXPORT_DIR / "grpo_step_records.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate exported policy rows without GPU dependencies")
    mode.add_argument("--export-verl-parquet", type=Path, help="Convert per-step JSONL to veRL training Parquet")
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warm-start-model", help="Full SFT model checkpoint to anchor GRPO")
    warmup.add_argument("--sft-warmup", action="store_true", help="Warm up for one epoch using high-reward references")
    parser.add_argument("--use-vllm", action="store_true", help="Enable colocated vLLM generation during training")
    args = parser.parse_args(argv)

    if args.export_verl_parquet is not None:
        rows = verl_dataset_items(load_grpo_dataset(args.dataset_path))
        try:  # pragma: no cover - optional GPU environment dependency
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Parquet conversion requires datasets in the veRL environment.") from exc
        args.export_verl_parquet.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows).to_parquet(str(args.export_verl_parquet))
        summary = {"dataset_records": len(rows), "parquet_path": str(args.export_verl_parquet), "status": "ok"}
    else:
        summary = run_training(
            dataset_path=args.dataset_path, model_name=args.model_name, output_dir=args.output_dir,
            learning_rate=args.learning_rate, num_generations=args.num_generations, max_steps=args.max_steps,
            dry_run=args.dry_run, warm_start_model=args.warm_start_model, sft_warmup=args.sft_warmup,
            use_vllm=args.use_vllm,
        )
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
