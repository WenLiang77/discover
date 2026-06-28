import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from university_gpu.local_hf_lora_model import (
    load_tokenizer_and_lora_model,
    make_optimizer,
)
from university_gpu.local_generation import (
    build_denoising_prompt,
    generate_candidate_code,
)
from university_gpu.local_reward_training import train_lora_on_rollouts


REPO_ROOT = Path(__file__).resolve().parents[1]


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_initial_magic_code():
    """
    Load the official MAGIC baseline code from examples.denoising.utils.

    This is the initial state for the denoising task.
    If the import fails, we use a very simple fallback function.
    """
    try:
        from examples.denoising.utils import MAGIC_FUNC

        return MAGIC_FUNC
    except Exception as error:
        print("Could not import MAGIC_FUNC from examples.denoising.utils.")
        print("Using fallback baseline.")
        print("Error:", repr(error))

        return """
def magic_denoise(X, **kwargs):
    import numpy as np
    X = np.asarray(X, dtype=float)
    return np.maximum(X, 0.0)
""".strip()


def evaluate_candidate_dummy(code: str):
    """
    Very fast fake evaluator.

    This is only for debugging the local TTT loop.
    It does not reproduce the paper's Section 4.4 denoising evaluator.

    Use this first on the server to check:
    - model loading
    - code generation
    - reward-weighted LoRA update
    """
    if "def magic_denoise" not in code:
        return {
            "ok": False,
            "mse": None,
            "poisson": None,
            "reward": -1.0,
            "error": "No magic_denoise function found.",
        }

    score = 1.0

    if "sqrt" in code:
        score += 0.2
    if "normalize" in code or "sum" in code:
        score += 0.2
    if "PCA" in code or "TruncatedSVD" in code or "svd" in code.lower():
        score += 0.2
    if "np.maximum" in code or "clip" in code:
        score += 0.1

    mse = 1.0 / score
    poisson = 1.0
    reward = 1.0 / mse

    return {
        "ok": True,
        "mse": mse,
        "poisson": poisson,
        "reward": reward,
        "error": None,
    }


def evaluate_candidate_official(code: str, timeout_seconds: int = 900):
    """
    Run the official Section 4.4 denoising evaluator in a subprocess.

    This uses:
        examples.denoising.utils.run_denoising_eval

    It may take much longer than dummy mode because it loads data and runs
    the actual single-cell denoising benchmark.
    """
    if "def magic_denoise" not in code:
        return {
            "ok": False,
            "mse": None,
            "poisson": None,
            "reward": -1.0,
            "error": "Generated code does not define magic_denoise.",
        }

    eval_program = f"""
import json
import math
import os
import sys
import traceback

sys.path.insert(0, {str(REPO_ROOT)!r})

import numpy as np

try:
    import scipy
except Exception:
    pass

try:
    import sklearn
except Exception:
    pass

try:
    import scanpy as sc
except Exception:
    pass

try:
    import scprep
except Exception:
    pass

try:
    import graphtools
except Exception:
    pass

from examples.denoising.utils import run_denoising_eval

{code}

try:
    if "magic_denoise" not in globals():
        raise RuntimeError("magic_denoise is not defined")

    mse, poisson = run_denoising_eval(magic_denoise, seed=42)

    mse = float(mse)
    poisson = float(poisson)

    if not math.isfinite(mse) or not math.isfinite(poisson):
        raise RuntimeError(f"Non-finite result: mse={{mse}}, poisson={{poisson}}")

    reward = 1.0 / max(mse, 1e-8)

    print("JSON_RESULT_START")
    print(json.dumps({{
        "ok": True,
        "mse": mse,
        "poisson": poisson,
        "reward": reward,
        "error": None,
    }}))
    print("JSON_RESULT_END")

except Exception as e:
    print("JSON_RESULT_START")
    print(json.dumps({{
        "ok": False,
        "mse": None,
        "poisson": None,
        "reward": -1.0,
        "error": repr(e),
        "traceback": traceback.format_exc(),
    }}))
    print("JSON_RESULT_END")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        eval_path = Path(tmpdir) / "eval_candidate.py"
        eval_path.write_text(eval_program, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        try:
            result = subprocess.run(
                [sys.executable, str(eval_path)],
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "mse": None,
                "poisson": None,
                "reward": -1.0,
                "error": f"Evaluation timeout after {timeout_seconds} seconds.",
            }

    stdout = result.stdout
    stderr = result.stderr

    if stderr:
        print("Evaluator STDERR tail:")
        print(stderr[-4000:])

    try:
        start = stdout.index("JSON_RESULT_START") + len("JSON_RESULT_START")
        end = stdout.index("JSON_RESULT_END")
        payload = stdout[start:end].strip()
        return json.loads(payload)
    except Exception:
        return {
            "ok": False,
            "mse": None,
            "poisson": None,
            "reward": -1.0,
            "error": "Could not parse evaluator output.",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }


def evaluate_candidate(code: str, eval_mode: str):
    """
    Choose between dummy evaluator and official evaluator.
    """
    if eval_mode == "dummy":
        return evaluate_candidate_dummy(code)

    if eval_mode == "official":
        return evaluate_candidate_official(code)

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        default=os.environ.get("LOCAL_TTT_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    )
    parser.add_argument(
        "--eval-mode",
        choices=["dummy", "official"],
        default=os.environ.get("LOCAL_TTT_EVAL_MODE", "dummy"),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("LOCAL_TTT_STEPS", "1")),
    )
    parser.add_argument(
        "--rollouts-per-step",
        type=int,
        default=int(os.environ.get("LOCAL_TTT_ROLLOUTS", "2")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("LOCAL_TTT_MAX_NEW_TOKENS", "1024")),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=int(os.environ.get("LOCAL_TTT_LORA_RANK", "8")),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.environ.get("LOCAL_TTT_LR", "1e-4")),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "LOCAL_TTT_OUTPUT_DIR",
            "university_gpu/outputs/local_ttt_denoising_tiny",
        ),
    )

    args = parser.parse_args()

    print_section("Tiny Local TTT-Denoising Experiment")
    print("Repo root:", REPO_ROOT)
    print("Model:", args.model_name)
    print("Eval mode:", args.eval_mode)
    print("Steps:", args.steps)
    print("Rollouts per step:", args.rollouts_per_step)
    print("Max new tokens:", args.max_new_tokens)
    print("LoRA rank:", args.lora_rank)
    print("Learning rate:", args.learning_rate)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("Load Local LLM + LoRA")
    tokenizer, model = load_tokenizer_and_lora_model(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
    )
    optimizer = make_optimizer(model, learning_rate=args.learning_rate)

    print_section("Load Initial MAGIC Baseline")
    current_code = load_initial_magic_code()
    print("Initial code preview:")
    print(current_code[:1200])

    history = []

    best_record = {
        "ok": False,
        "mse": None,
        "poisson": None,
        "reward": -1.0,
        "code": current_code,
        "source": "initial",
    }

    for step in range(1, args.steps + 1):
        print_section(f"Step {step}/{args.steps}")

        prompt = build_denoising_prompt(
            current_code=current_code,
            history=history,
        )

        rollout_prompts = []
        rollout_responses = []
        rollout_rewards = []
        step_records = []

        for rollout_id in range(1, args.rollouts_per_step + 1):
            print_section(f"Step {step} Rollout {rollout_id}/{args.rollouts_per_step}")

            start_time = time.time()

            generation = generate_candidate_code(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
            )

            code = generation["code"]

            print("Generated code preview:")
            print(code[:1200])

            eval_result = evaluate_candidate(
                code=code,
                eval_mode=args.eval_mode,
            )

            elapsed = time.time() - start_time

            record = {
                "step": step,
                "rollout": rollout_id,
                "ok": eval_result.get("ok"),
                "mse": eval_result.get("mse"),
                "poisson": eval_result.get("poisson"),
                "reward": eval_result.get("reward"),
                "error": eval_result.get("error"),
                "elapsed_seconds": elapsed,
                "code": code,
                "raw_text": generation["raw_text"],
            }

            history.append(record)
            step_records.append(record)

            rollout_prompts.append(generation["rendered_prompt"])
            rollout_responses.append(code)
            rollout_rewards.append(float(eval_result.get("reward", -1.0)))

            print("Evaluation result:")
            print(json.dumps(eval_result, indent=2))
            print(f"Elapsed seconds: {elapsed:.2f}")

            if record["ok"] and record["reward"] > best_record["reward"]:
                best_record = {
                    **record,
                    "source": f"step_{step}_rollout_{rollout_id}",
                }
                current_code = code

        print_section("LoRA Update")
        train_result = train_lora_on_rollouts(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            prompts=rollout_prompts,
            responses=rollout_responses,
            rewards=rollout_rewards,
            beta=2.0,
            max_length=4096,
        )

        print("Train result:")
        print(json.dumps(train_result, indent=2))

        save_json(
            output_dir / "history.json",
            {
                "args": vars(args),
                "history": history,
                "best_record": best_record,
            },
        )

        adapter_dir = output_dir / f"lora_adapter_step_{step}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)

        print("Saved LoRA adapter to:", adapter_dir)

    print_section("Final Best Record")
    summary = {
        "ok": best_record.get("ok"),
        "mse": best_record.get("mse"),
        "poisson": best_record.get("poisson"),
        "reward": best_record.get("reward"),
        "source": best_record.get("source"),
    }
    print(json.dumps(summary, indent=2))

    best_code_path = output_dir / "best_magic_denoise.py"
    best_code_path.write_text(best_record["code"], encoding="utf-8")
    print("Saved best code to:", best_code_path)

    save_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()