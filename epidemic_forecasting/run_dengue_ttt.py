from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from epidemic_forecasting.core.generation import (
    build_task_prompt,
    generate_candidate_code,
)
from epidemic_forecasting.core.local_hf_lora_model import (
    load_tokenizer_and_lora_model,
    make_optimizer,
)
from epidemic_forecasting.core.local_reward_training import (
    train_lora_on_rollouts,
)
from epidemic_forecasting.core.puct_sampler import (
    PUCTSampler,
    SearchState,
)
from epidemic_forecasting.tasks.dengue.task import (
    create_dengue_task,
)


DEFAULT_MODEL_NAME = os.environ.get(
    "LOCAL_TTT_MODEL",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def result_to_dict(result) -> dict[str, Any]:
    return json_safe(
        {
            "ok": result.ok,
            "reward": result.reward,
            "metrics": result.metrics,
            "error": result.error,
            "behavior_signature": result.behavior_signature,
            "metadata": result.metadata,
        }
    )


def make_output_directory(
    requested_path: str | None,
    task_id: str,
) -> Path:
    if requested_path:
        output_directory = Path(requested_path).expanduser().resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_directory = (
            Path(__file__).resolve().parent
            / "results"
            / "ttt"
            / task_id
            / timestamp
        )

    output_directory.mkdir(parents=True, exist_ok=False)
    return output_directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_observation(result, duplicate: bool) -> str:
    if not result.ok:
        return f"Invalid candidate: {result.error}"

    metric_text = ", ".join(
        f"{name}={value:.8g}"
        for name, value in sorted(result.metrics.items())
    )
    duplicate_text = " duplicate_behavior=True" if duplicate else ""
    return f"reward={result.reward:.8g}; {metric_text};{duplicate_text}"


def history_item(
    result,
    duplicate: bool,
    adjusted_reward: float | None,
) -> dict[str, Any]:
    item = result.to_history_item()
    item["duplicate_valid_behavior"] = duplicate
    item["raw_reward"] = result.reward
    item["reward"] = adjusted_reward
    return item


def save_rollout(
    rollout_directory: Path,
    generated: dict[str, str] | None,
    payload: dict[str, Any],
    result,
) -> None:
    rollout_directory.mkdir(parents=True, exist_ok=False)

    if generated is not None:
        (rollout_directory / "rendered_prompt.txt").write_text(
            generated.get("rendered_prompt", ""),
            encoding="utf-8",
        )
        (rollout_directory / "raw_response.txt").write_text(
            generated.get("raw_text", ""),
            encoding="utf-8",
        )
        (rollout_directory / "candidate.py").write_text(
            generated.get("code", ""),
            encoding="utf-8",
        )

    if result is not None and result.predictions is not None:
        np.save(rollout_directory / "predictions.npy", result.predictions)

    write_json(rollout_directory / "result.json", payload)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.parents_per_step < 1:
        raise ValueError("--parents-per-step must be at least 1.")
    if args.rollouts_per_parent < 2 and not args.dry_run:
        raise ValueError(
            "--rollouts-per-parent must be at least 2 so that "
            "reward advantages can be computed."
        )
    if args.runtime_budget_seconds <= 0:
        raise ValueError("--runtime-budget-seconds must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.lora_rank < 1:
        raise ValueError("--lora-rank must be at least 1.")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1).")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if args.max_input_tokens < 1:
        raise ValueError("--max-input-tokens must be positive.")
    if args.training_max_length < 1:
        raise ValueError("--training-max-length must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.max_buffer_size < 1:
        raise ValueError("--max-buffer-size must be at least 1.")
    if args.topk_children < 1:
        raise ValueError("--topk-children must be at least 1.")
    if args.duplicate_penalty < 0:
        raise ValueError("--duplicate-penalty must not be negative.")


def run_ttt(args: argparse.Namespace) -> Path:
    set_seed(args.seed)

    task = create_dengue_task(
        dataset=args.dataset,
        forecast_horizon=args.forecast_horizon,
        runtime_budget_seconds=args.runtime_budget_seconds,
        random_state=args.seed,
        mase_seasonality=args.mase_seasonality,
    )
    output_directory = make_output_directory(args.output_dir, task.config.task_id)

    print("=" * 80)
    print("Local TTT epidemic forecasting")
    print("=" * 80)
    print("Output directory:", output_directory)
    print("Task:", task.describe())

    initial_state = SearchState(
        timestep=0,
        code="",
        value=None,
        metrics={},
        observation="No model-generated candidate exists yet.",
        metadata={
            "source": "generation_root",
            "task_id": task.config.task_id,
        },
        behavior_signature=None,
    )
    sampler = PUCTSampler(
        file_path=output_directory / "puct" / "sampler.json",
        initial_state_factory=lambda: initial_state,
        max_buffer_size=args.max_buffer_size,
        batch_size=1,
        puct_c=args.puct_c,
        topk_children=args.topk_children,
        resume=False,
    )

    run_config = {
        "runner": "local_ttt",
        "task": task.describe(),
        "model_name": None if args.dry_run else args.model_name,
        "steps": 0 if args.dry_run else args.steps,
        "parents_per_step": args.parents_per_step,
        "rollouts_per_parent": args.rollouts_per_parent,
        "maximum_generated_candidates": (
            0
            if args.dry_run
            else args.steps
            * args.parents_per_step
            * args.rollouts_per_parent
        ),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "training_max_length": args.training_max_length,
        "runtime_budget_seconds": args.runtime_budget_seconds,
        "mase_seasonality": args.mase_seasonality,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "puct_c": args.puct_c,
        "topk_children": args.topk_children,
        "max_buffer_size": args.max_buffer_size,
        "invalid_reward": args.invalid_reward,
        "duplicate_penalty": args.duplicate_penalty,
        "dry_run": args.dry_run,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_directory / "run_config.json", run_config)

    if args.dry_run:
        summary = {
            "schema_version": 2,
            "status": "dry_run_complete",
            "runner": "local_ttt",
            "task": task.describe(),
            "steps": [],
            "valid_generated_count": 0,
            "invalid_generated_count": 0,
            "duplicate_behavior_count": 0,
            "best_generated_state": None,
            "best_search_states": [],
            "best_model_result": None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_directory / "summary.json", summary)
        print("TTT dry run completed successfully.")
        return output_directory

    tokenizer, model = load_tokenizer_and_lora_model(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    optimizer = make_optimizer(model=model, learning_rate=args.learning_rate)

    all_history: list[dict[str, Any]] = []
    seen_behavior_signatures: dict[str, str] = {}

    step_summaries: list[dict[str, Any]] = []
    total_valid = 0
    total_invalid = 0
    total_duplicates = 0
    global_rollout_index = 0
    best_generated: dict[str, Any] | None = None

    for step in range(1, args.steps + 1):
        print()
        print("=" * 80)
        print(f"TTT step {step}/{args.steps}")
        print("=" * 80)

        step_directory = output_directory / f"step_{step:04d}"
        step_directory.mkdir(parents=True, exist_ok=False)

        parent_states = sampler.sample_states(args.parents_per_step)
        step_payloads: list[dict[str, Any]] = []
        step_valid_children: list[SearchState] = []
        step_valid_parents: list[SearchState] = []
        training_prompts: list[str] = []
        training_responses: list[str] = []
        training_rewards: list[float] = []

        for parent_number, parent in enumerate(parent_states, start=1):
            prompt = build_task_prompt(
                prompt_path=task.config.prompt_path,
                current_code=parent.code,
                history=all_history,
            )

            for rollout_number in range(1, args.rollouts_per_parent + 1):
                global_rollout_index += 1
                rollout_seed = args.seed + global_rollout_index
                set_seed(rollout_seed)

                rollout_directory = (
                    step_directory
                    / (
                        f"parent_{parent_number:02d}_"
                        f"rollout_{rollout_number:02d}"
                    )
                )

                generated: dict[str, str] | None = None
                evaluation_result = None
                started = time.perf_counter()

                duplicate_behavior = False
                duplicate_of = None
                raw_reward = None
                adjusted_reward = float(args.invalid_reward)

                try:
                    generated = generate_candidate_code(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        function_name=task.config.function_name,
                        max_new_tokens=args.max_new_tokens,
                        max_input_tokens=args.max_input_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                    candidate_code = generated.get("code", "").strip()
                    if not candidate_code:
                        raise ValueError(
                            "The model response did not contain candidate code."
                        )

                    evaluation_result = task.evaluate_code(candidate_code)

                    if (
                        evaluation_result.ok
                        and evaluation_result.reward is not None
                    ):
                        total_valid += 1
                        raw_reward = float(evaluation_result.reward)
                        adjusted_reward = raw_reward

                        signature = evaluation_result.behavior_signature
                        if signature:
                            duplicate_of = seen_behavior_signatures.get(signature)
                            duplicate_behavior = duplicate_of is not None

                            if duplicate_behavior:
                                total_duplicates += 1
                                adjusted_reward -= args.duplicate_penalty
                            else:
                                seen_behavior_signatures[signature] = (
                                    f"step_{step:04d}/"
                                    f"parent_{parent_number:02d}_"
                                    f"rollout_{rollout_number:02d}"
                                )

                        child = SearchState(
                            timestep=step,
                            code=candidate_code,
                            value=float(adjusted_reward),
                            metrics=dict(evaluation_result.metrics),
                            observation=format_observation(
                                evaluation_result,
                                duplicate_behavior,
                            ),
                            metadata={
                                "step": step,
                                "parent_number": parent_number,
                                "rollout_number": rollout_number,
                                "raw_reward": raw_reward,
                                "adjusted_reward": adjusted_reward,
                                "duplicate_behavior": duplicate_behavior,
                                "duplicate_of": duplicate_of,
                                "evaluation_metadata": (
                                    evaluation_result.metadata
                                ),
                            },
                            behavior_signature=signature,
                        )
                        step_valid_children.append(child)
                        step_valid_parents.append(parent)

                        if (
                            best_generated is None
                            or raw_reward > best_generated["raw_reward"]
                        ):
                            best_generated = {
                                "step": step,
                                "parent_number": parent_number,
                                "rollout_number": rollout_number,
                                "raw_reward": raw_reward,
                                "adjusted_reward": adjusted_reward,
                                "metrics": dict(evaluation_result.metrics),
                                "duplicate_behavior": duplicate_behavior,
                                "candidate_path": str(
                                    rollout_directory / "candidate.py"
                                ),
                            }
                    else:
                        total_invalid += 1
                        sampler.record_failed_rollout(parent, save=False)

                except Exception as error:
                    total_invalid += 1
                    sampler.record_failed_rollout(parent, save=False)
                    error_message = (
                        "Generation failed: "
                        f"{type(error).__name__}: {error}"
                    )

                    from epidemic_forecasting.tasks.base import EvaluationResult

                    evaluation_result = EvaluationResult(
                        ok=False,
                        error=error_message,
                        metadata={"stage": "generation"},
                    )

                elapsed_seconds = time.perf_counter() - started

                if generated is not None:
                    response_text = generated.get("raw_text", "")
                    rendered_prompt = generated.get(
                        "rendered_prompt",
                        prompt,
                    )
                    if response_text.strip():
                        training_prompts.append(rendered_prompt)
                        training_responses.append(response_text)
                        training_rewards.append(float(adjusted_reward))

                all_history.append(
                    history_item(
                        evaluation_result,
                        duplicate=duplicate_behavior,
                        adjusted_reward=float(adjusted_reward),
                    )
                )

                rollout_payload = {
                    "step": step,
                    "parent_number": parent_number,
                    "parent_state_id": parent.id,
                    "parent_reward": parent.value,
                    "rollout_number": rollout_number,
                    "global_rollout_index": global_rollout_index,
                    "seed": rollout_seed,
                    "elapsed_seconds": elapsed_seconds,
                    "raw_reward": raw_reward,
                    "adjusted_reward": adjusted_reward,
                    "duplicate_valid_behavior": duplicate_behavior,
                    "duplicate_of": duplicate_of,
                    "evaluation": result_to_dict(evaluation_result),
                }

                save_rollout(
                    rollout_directory=rollout_directory,
                    generated=generated,
                    payload=rollout_payload,
                    result=evaluation_result,
                )
                step_payloads.append(rollout_payload)

                print(
                    f"Parent {parent_number}, rollout {rollout_number}: "
                    f"valid={evaluation_result.ok}, "
                    f"raw_reward={raw_reward}, "
                    f"adjusted_reward={adjusted_reward}, "
                    f"duplicate={duplicate_behavior}"
                )

        if step_valid_children:
            sampler.update_states(
                states=step_valid_children,
                parent_states=step_valid_parents,
                save=False,
                step=step,
            )

        if len(training_rewards) >= 2:
            training_result = train_lora_on_rollouts(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                prompts=training_prompts,
                responses=training_responses,
                rewards=training_rewards,
                max_length=args.training_max_length,
            )
        else:
            training_result = {
                "ok": False,
                "loss": None,
                "message": (
                    "Fewer than two generated responses were "
                    "available for training."
                ),
                "rewards": training_rewards,
            }

        sampler.flush(step=step)

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        step_summary = {
            "step": step,
            "parents": [parent.to_dict() for parent in parent_states],
            "rollouts": step_payloads,
            "training": training_result,
            "puct_stats": sampler.get_sample_stats(),
            "best_states": [
                state.to_dict()
                for state in sampler.best_states(limit=5)
            ],
        }
        write_json(step_directory / "step_summary.json", step_summary)
        step_summaries.append(step_summary)

        print("Training:", training_result)
        print("PUCT stats:", sampler.get_sample_stats())

    best_state_payloads = [
        state.to_dict()
        for state in sampler.best_states(limit=10)
    ]

    if best_generated is not None:
        best_candidate_path = Path(best_generated["candidate_path"])
        if best_candidate_path.is_file():
            (
                output_directory / "best_generated_candidate.py"
            ).write_text(
                best_candidate_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    if args.save_adapter:
        adapter_directory = output_directory / "lora_adapter"
        model.save_pretrained(adapter_directory)
        tokenizer.save_pretrained(adapter_directory)

    summary = {
        "schema_version": 2,
        "status": "complete",
        "runner": "local_ttt",
        "task": task.describe(),
        "steps": step_summaries,
        "valid_generated_count": total_valid,
        "invalid_generated_count": total_invalid,
        "duplicate_behavior_count": total_duplicates,
        "best_generated_state": best_generated,
        "best_search_states": best_state_payloads,
        "best_model_result": best_generated,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_directory / "summary.json", summary)
    write_json(output_directory / "history.json", {"history": all_history})

    print()
    print("=" * 80)
    print("TTT experiment complete")
    print("=" * 80)
    print("Valid generated candidates:", total_valid)
    print("Invalid generated candidates:", total_invalid)
    print("Duplicate behaviours:", total_duplicates)
    print("Best generated:", best_generated)
    print("Best model-generated result:", best_generated)
    print("Results:", output_directory)

    return output_directory


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run local LoRA + PUCT test-time training for "
            "Dengue epidemic forecasting."
        )
    )

    parser.add_argument("--dataset", choices=("colombia", "panama"), default="colombia")
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        choices=(8,),
        default=8,
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--parents-per-step", type=int, default=1)
    parser.add_argument("--rollouts-per-parent", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--training-max-length", type=int, default=4096)
    parser.add_argument(
        "--runtime-budget-seconds",
        type=float,
        default=400.0,
    )
    parser.add_argument("--mase-seasonality", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--topk-children", type=int, default=2)
    parser.add_argument("--max-buffer-size", type=int, default=1000)
    parser.add_argument("--invalid-reward", type=float, default=-250.0)
    parser.add_argument(
        "--duplicate-penalty",
        type=float,
        default=0.0,
        help=(
            "Optional penalty subtracted from duplicate valid "
            "forecast rewards. Keep at 0 for the primary experiment."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--save-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check the task, prompt, output and PUCT setup "
            "without loading the language model."
        ),
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)
    run_ttt(args)


if __name__ == "__main__":
    main()
