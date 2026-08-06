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

from epidemic_forecasting.tasks.covid19.task import (
    create_covid19_task,
)


DEFAULT_MODEL_NAME = os.environ.get(
    "LOCAL_TTT_MODEL",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
)


def json_safe(value: Any) -> Any:
    """Convert nested NumPy and pathlib values into JSON-safe objects."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one UTF-8 JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def result_to_dict(result, include_predictions: bool = False) -> dict[str, Any]:
    """Convert EvaluationResult into a compact serialisable dictionary."""
    payload = {
        "ok": result.ok,
        "reward": result.reward,
        "metrics": result.metrics,
        "error": result.error,
        "behavior_signature": result.behavior_signature,
        "metadata": result.metadata,
    }

    if include_predictions and result.predictions is not None:
        payload["predictions"] = result.predictions

    return json_safe(payload)


def make_output_directory(
    requested_path: str | None,
    task_id: str,
) -> Path:
    """Create a unique directory for one baseline experiment."""
    if requested_path:
        output_directory = Path(requested_path).expanduser().resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        output_directory = (
            Path(__file__).resolve().parent
            / "results"
            / "baseline"
            / task_id
            / timestamp
        )

    output_directory.mkdir(parents=True, exist_ok=False)
    return output_directory


def set_generation_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before one generation."""
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_inference_model(model_name: str):
    """
    Load a base causal language model without LoRA adapters.

    This is intentionally separate from the TTT model loader: the baseline
    must measure independent generations from the unchanged base model.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    print("=" * 80)
    print("Loading inference-only baseline model")
    print("=" * 80)
    print("Model:", model_name)
    print("Compute dtype:", compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    model.config.use_cache = True

    return tokenizer, model


def build_fixed_baseline_prompt(task) -> str:
    """
    Build one fixed prompt for independent no-TTT generations.

    No hand-written forecasting implementation or hand-written result is
    supplied to the model. Every attempt starts from the same task-only prompt.
    """
    from epidemic_forecasting.core.generation import (
        build_task_prompt,
    )

    return build_task_prompt(
        prompt_path=task.config.prompt_path,
        current_code="",
        history=None,
    )


def save_attempt(
    attempt_directory: Path,
    attempt_payload: dict[str, Any],
    generated: dict[str, str] | None,
    evaluation_result,
) -> None:
    """Save the prompt, response, code, metrics, and optional predictions."""
    attempt_directory.mkdir(parents=True, exist_ok=False)

    if generated is not None:
        (attempt_directory / "rendered_prompt.txt").write_text(
            generated.get("rendered_prompt", ""),
            encoding="utf-8",
        )
        (attempt_directory / "raw_response.txt").write_text(
            generated.get("raw_text", ""),
            encoding="utf-8",
        )
        (attempt_directory / "candidate.py").write_text(
            generated.get("code", ""),
            encoding="utf-8",
        )

    if (
        evaluation_result is not None
        and evaluation_result.predictions is not None
    ):
        np.save(
            attempt_directory / "predictions.npy",
            evaluation_result.predictions,
        )

    write_json(
        attempt_directory / "result.json",
        attempt_payload,
    )


def run_baseline(args: argparse.Namespace) -> Path:
    """Run an independent Qwen-only code-generation baseline."""
    task = create_covid19_task(
        dataset=args.dataset,
        forecast_horizon=args.forecast_horizon,
        runtime_budget_seconds=args.runtime_budget_seconds,
        random_state=args.seed,
        mase_seasonality=args.mase_seasonality,
    )

    output_directory = make_output_directory(
        requested_path=args.output_dir,
        task_id=task.config.task_id,
    )

    print("=" * 80)
    print("COVID-19 Qwen-only baseline")
    print("=" * 80)
    print("Output directory:", output_directory)
    print("Task:", task.describe())

    run_configuration = {
        "runner": "qwen_only_baseline",
        "task": task.describe(),
        "model_name": None if args.dry_run else args.model_name,
        "attempts": 0 if args.dry_run else args.attempts,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "runtime_budget_seconds": args.runtime_budget_seconds,
        "mase_seasonality": args.mase_seasonality,
        "dry_run": args.dry_run,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(
        output_directory / "run_config.json",
        run_configuration,
    )

    fixed_prompt = build_fixed_baseline_prompt(task=task)

    if args.dry_run:
        summary = {
            "schema_version": 2,
            "status": "dry_run_complete",
            "runner": "qwen_only_baseline",
            "task": task.describe(),
            "generated_attempts": [],
            "valid_generated_count": 0,
            "invalid_generated_count": 0,
            "duplicate_behavior_count": 0,
            "best_generated_attempt": None,
            "best_model_result": None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_directory / "summary.json", summary)
        print("No-TTT dry run completed successfully.")
        return output_directory

    from epidemic_forecasting.core.generation import (
        generate_candidate_code,
    )

    tokenizer, model = load_inference_model(args.model_name)
    attempt_summaries: list[dict[str, Any]] = []
    seen_behavior_signatures: dict[str, str] = {}

    best_generated: dict[str, Any] | None = None

    for attempt_index in range(1, args.attempts + 1):
        attempt_seed = args.seed + attempt_index - 1
        set_generation_seed(attempt_seed)

        print()
        print("-" * 80)
        print(
            f"Attempt {attempt_index}/{args.attempts} "
            f"(seed={attempt_seed})"
        )
        print("-" * 80)

        attempt_directory = (
            output_directory
            / f"attempt_{attempt_index:04d}"
        )

        generated: dict[str, str] | None = None
        evaluation_result = None
        started = time.perf_counter()

        try:
            generated = generate_candidate_code(
                model=model,
                tokenizer=tokenizer,
                prompt=fixed_prompt,
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

            duplicate_of = None
            duplicate_behavior = False

            if (
                evaluation_result.ok
                and evaluation_result.behavior_signature
            ):
                duplicate_of = seen_behavior_signatures.get(
                    evaluation_result.behavior_signature
                )
                duplicate_behavior = duplicate_of is not None

                if not duplicate_behavior:
                    seen_behavior_signatures[
                        evaluation_result.behavior_signature
                    ] = f"attempt_{attempt_index:04d}"

            elapsed_seconds = time.perf_counter() - started

            attempt_payload = {
                "attempt": attempt_index,
                "seed": attempt_seed,
                "elapsed_seconds": elapsed_seconds,
                "duplicate_valid_behavior": duplicate_behavior,
                "duplicate_of": duplicate_of,
                "evaluation": result_to_dict(evaluation_result),
            }

            if (
                evaluation_result.ok
                and evaluation_result.reward is not None
                and (
                    best_generated is None
                    or evaluation_result.reward
                    > best_generated["reward"]
                )
            ):
                best_generated = {
                    "attempt": attempt_index,
                    "reward": float(evaluation_result.reward),
                    "metrics": dict(evaluation_result.metrics),
                    "duplicate_valid_behavior": duplicate_behavior,
                    "candidate_path": str(
                        attempt_directory / "candidate.py"
                    ),
                }

            print("Valid:", evaluation_result.ok)
            print("Reward:", evaluation_result.reward)
            print("Metrics:", evaluation_result.metrics)
            print("Duplicate behavior:", duplicate_behavior)
            print("Error:", evaluation_result.error)

        except Exception as error:
            elapsed_seconds = time.perf_counter() - started
            attempt_payload = {
                "attempt": attempt_index,
                "seed": attempt_seed,
                "elapsed_seconds": elapsed_seconds,
                "duplicate_valid_behavior": False,
                "duplicate_of": None,
                "evaluation": {
                    "ok": False,
                    "reward": None,
                    "metrics": {},
                    "error": (
                        f"Generation failed: "
                        f"{type(error).__name__}: {error}"
                    ),
                    "behavior_signature": None,
                    "metadata": {
                        "stage": "generation",
                    },
                },
            }

            print(attempt_payload["evaluation"]["error"])

        save_attempt(
            attempt_directory=attempt_directory,
            attempt_payload=attempt_payload,
            generated=generated,
            evaluation_result=evaluation_result,
        )
        attempt_summaries.append(attempt_payload)

    valid_attempts = [
        item
        for item in attempt_summaries
        if item["evaluation"]["ok"]
    ]
    invalid_attempts = [
        item
        for item in attempt_summaries
        if not item["evaluation"]["ok"]
    ]
    duplicate_count = sum(
        bool(item["duplicate_valid_behavior"])
        for item in attempt_summaries
    )

    if best_generated is not None:
        best_candidate_path = Path(best_generated["candidate_path"])
        if best_candidate_path.is_file():
            (
                output_directory / "best_generated_candidate.py"
            ).write_text(
                best_candidate_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    summary = {
        "schema_version": 2,
        "status": "complete",
        "runner": "qwen_only_baseline",
        "task": task.describe(),
        "generated_attempts": attempt_summaries,
        "valid_generated_count": len(valid_attempts),
        "invalid_generated_count": len(invalid_attempts),
        "duplicate_behavior_count": duplicate_count,
        "best_generated_attempt": best_generated,
        "best_model_result": best_generated,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_directory / "summary.json", summary)

    print()
    print("=" * 80)
    print("Baseline experiment complete")
    print("=" * 80)
    print("Valid generated attempts:", len(valid_attempts))
    print("Invalid generated attempts:", len(invalid_attempts))
    print("Duplicate behaviours:", duplicate_count)
    print("Best generated:", best_generated)
    print("Best model-generated result:", best_generated)
    print("Results:", output_directory)

    return output_directory


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent Qwen-only code-generation baselines for the "
            "COVID-19 epidemic forecasting task."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=("uk", "us"),
        default="uk",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        choices=(7, 14, 30),
        default=14,
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1536,
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--runtime-budget-seconds",
        type=float,
        default=400.0,
    )
    parser.add_argument(
        "--mase-seasonality",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate task, prompt and output setup without loading Qwen."
        ),
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.attempts < 1 and not args.dry_run:
        raise ValueError("--attempts must be at least 1.")

    if args.runtime_budget_seconds <= 0:
        raise ValueError(
            "--runtime-budget-seconds must be positive."
        )

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")

    if args.max_input_tokens < 1:
        raise ValueError("--max-input-tokens must be positive.")

    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")

    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in the interval (0, 1].")

    if args.mase_seasonality < 1:
        raise ValueError("--mase-seasonality must be at least 1.")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)
    run_baseline(args)


if __name__ == "__main__":
    main()
