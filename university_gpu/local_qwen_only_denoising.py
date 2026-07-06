"""
Qwen-only denoising baseline.

This script uses the same HuggingFace model and the same official denoising
evaluator, but does not use TTT:
- no PUCT state reuse
- no LoRA training
- no reward-based model update

It repeatedly samples candidate magic_denoise functions from the base model and
keeps the best valid candidate under the official denoising evaluator.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from university_gpu.local_puct_sampler import DEFAULT_MAGIC_FUNC
from university_gpu.local_generation import build_denoising_prompt, generate_candidate_code
from university_gpu.local_ttt_denoising_tiny import evaluate_candidate


REPO_ROOT = Path(__file__).resolve().parents[1]


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_compute_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_base_model(model_name: str):
    print(f"Loading base model without LoRA: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_compute_dtype()
    print(f"Using dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return tokenizer, model


def set_generation_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("LOCAL_TTT_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="official",
        choices=["official", "dummy"],
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=int(os.environ.get("LOCAL_QWEN_ONLY_ATTEMPTS", "50")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("LOCAL_TTT_MAX_NEW_TOKENS", "1536")),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("LOCAL_QWEN_ONLY_TEMPERATURE", "0.7")),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.environ.get("LOCAL_QWEN_ONLY_TOP_P", "0.95")),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("LOCAL_QWEN_ONLY_SEED", "42")),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get(
            "LOCAL_QWEN_ONLY_OUTPUT_DIR",
            "university_gpu/outputs/qwen_only_denoising",
        ),
    )

    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Qwen-only denoising baseline ===")
    print(f"Model: {args.model_name}")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Attempts: {args.attempts}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {output_dir}")

    tokenizer, model = load_base_model(args.model_name)

    history = []
    best_record = None

    for attempt in range(1, args.attempts + 1):
        print(f"\n=== Attempt {attempt}/{args.attempts} ===")

        # Qwen-only baseline:
        # every attempt starts from the same initial implementation.
        # no previous history, no PUCT, no LoRA update.
        prompt = build_denoising_prompt(
            current_code=DEFAULT_MAGIC_FUNC,
            history=[],
        )

        set_generation_seed(args.seed + attempt)

        start_time = time.time()

        generation = generate_candidate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        code = generation["code"]

        eval_result = evaluate_candidate(
            code=code,
            eval_mode=args.eval_mode,
        )

        elapsed = time.time() - start_time

        record = {
            "attempt": attempt,
            "ok": eval_result.get("ok"),
            "mse": eval_result.get("mse"),
            "poisson": eval_result.get("poisson"),
            "reward": eval_result.get("reward"),
            "raw_score": eval_result.get("raw_score"),
            "metrics": eval_result.get("metrics", {}),
            "error": eval_result.get("error"),
            "elapsed_seconds": elapsed,
            "code": code,
            "raw_text": generation["raw_text"],
            "rendered_prompt": generation["rendered_prompt"],
            "method": "qwen_only_no_ttt",
        }

        history.append(record)

        print(
            "ok={ok} mse={mse} poisson={poisson} reward={reward} error={error}".format(
                ok=record["ok"],
                mse=record["mse"],
                poisson=record["poisson"],
                reward=record["reward"],
                error=record["error"],
            )
        )

        if record["ok"]:
            if best_record is None or record["reward"] > best_record["reward"]:
                best_record = {
                    **record,
                    "source": f"attempt_{attempt}",
                }
                print(f"New best valid candidate: {best_record['source']}")

        valid_count = sum(1 for item in history if item.get("ok"))

        save_json(
            output_dir / "history.json",
            {
                "args": vars(args),
                "history": history,
                "best_record": best_record,
                "total_candidates": len(history),
                "valid_candidates": valid_count,
                "valid_rate": valid_count / max(len(history), 1),
                "method": "qwen_only_no_ttt",
            },
        )

    valid_count = sum(1 for item in history if item.get("ok"))

    if best_record is None:
        summary = {
            "ok": False,
            "method": "qwen_only_no_ttt",
            "total_candidates": len(history),
            "valid_candidates": valid_count,
            "valid_rate": valid_count / max(len(history), 1),
            "best_mse": None,
            "best_poisson": None,
            "best_reward": None,
            "source": None,
        }
    else:
        summary = {
            "ok": True,
            "method": "qwen_only_no_ttt",
            "total_candidates": len(history),
            "valid_candidates": valid_count,
            "valid_rate": valid_count / max(len(history), 1),
            "best_mse": best_record.get("mse"),
            "best_poisson": best_record.get("poisson"),
            "best_reward": best_record.get("reward"),
            "best_raw_score": best_record.get("raw_score"),
            "best_metrics": best_record.get("metrics", {}),
            "source": best_record.get("source"),
        }

        best_code_path = output_dir / "best_magic_denoise.py"
        best_code_path.write_text(best_record["code"], encoding="utf-8")

    save_json(output_dir / "summary.json", summary)

    print("\n=== Final summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()