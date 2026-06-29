import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ttt_discover.tinker_utils.sampler import PUCTSampler
from examples.denoising.env import DenoisingState

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


class LocalDenoisingEnvForPUCT:
    """
    Minimal environment wrapper for official PUCTSampler.

    We use the official DenoisingState class, but we define the initial
    state value as -MSE because PUCT assumes higher value is better.

    Denoising minimizes MSE, so:
        value = -mse
    """

    state_type = DenoisingState

    @classmethod
    def create_initial_state(cls, problem_type: str):
        from examples.denoising.utils import MAGIC_FUNC

        initial_mse = 0.2316
        initial_poisson = 0.0370

        return DenoisingState(
            timestep=-1,
            construction=[],
            code=MAGIC_FUNC,
            value=-initial_mse,
            mse=initial_mse,
            poisson=initial_poisson,
        )


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def evaluate_candidate_dummy(code: str):
    """
    Very fast fake evaluator.

    This is only for debugging the local TTT loop.
    It does not reproduce the paper's Section 4.4 denoising evaluator.
    """
    if "def magic_denoise" not in code:
        return {
            "ok": False,
            "mse": None,
            "poisson": None,
            "reward": -1.0,
            "raw_score": None,
            "metrics": {},
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
        "raw_score": mse,
        "metrics": {
            "mse": mse,
            "poisson": poisson,
            "mse_normalized": None,
            "poisson_normalized": None,
        },
        "error": None,
    }


def evaluate_candidate_official(code: str, timeout_seconds: int = 900):
    """
    Run the official Section 4.4 denoising evaluator in a subprocess.

    This uses:
        examples.denoising.utils.run_denoising_eval

    Then it applies the official validity logic from examples.denoising.env:
        verify_denoising((mse, poisson))
    """
    if "def magic_denoise" not in code:
        return {
            "ok": False,
            "mse": None,
            "poisson": None,
            "reward": -1.0,
            "raw_score": None,
            "metrics": {},
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

    print("JSON_RESULT_START")
    print(json.dumps({{
        "ok": True,
        "mse": mse,
        "poisson": poisson,
        "error": None
    }}))
    print("JSON_RESULT_END")

except Exception as e:
    print("JSON_RESULT_START")
    print(json.dumps({{
        "ok": False,
        "mse": None,
        "poisson": None,
        "error": repr(e),
        "traceback": traceback.format_exc()
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
                "raw_score": None,
                "metrics": {},
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
        parsed = json.loads(payload)
    except Exception:
        return {
            "ok": False,
            "mse": None,
            "poisson": None,
            "reward": -1.0,
            "raw_score": None,
            "metrics": {},
            "error": "Could not parse evaluator output.",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }

    if not parsed.get("ok"):
        return {
            "ok": False,
            "mse": None,
            "poisson": None,
            "reward": -1.0,
            "raw_score": None,
            "metrics": {},
            "error": parsed.get("error"),
            "traceback": parsed.get("traceback"),
        }

    mse = float(parsed["mse"])
    poisson = float(parsed["poisson"])

    try:
        from examples.denoising.env import verify_denoising, BASELINES
    except Exception as error:
        return {
            "ok": False,
            "mse": mse,
            "poisson": poisson,
            "reward": -1.0,
            "raw_score": mse,
            "metrics": {},
            "error": f"Could not import official verify_denoising: {repr(error)}",
        }

    is_valid = verify_denoising((mse, poisson))

    baseline = BASELINES["pancreas"]

    mse_range = baseline["baseline_mse"] - baseline["perfect_mse"]
    poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]

    mse_normalized = None
    poisson_normalized = None

    if mse_range > 0:
        mse_normalized = (baseline["baseline_mse"] - mse) / mse_range
        mse_normalized = max(0.0, min(1.0, mse_normalized))

    if poisson_range > 0:
        poisson_normalized = (baseline["baseline_poisson"] - poisson) / poisson_range
        poisson_normalized = max(0.0, min(1.0, poisson_normalized))

    metrics = {
        "mse": mse,
        "poisson": poisson,
        "mse_normalized": mse_normalized,
        "poisson_normalized": poisson_normalized,
    }

    if not is_valid:
        return {
            "ok": False,
            "mse": mse,
            "poisson": poisson,
            "reward": -1.0,
            "raw_score": mse,
            "metrics": metrics,
            "error": "Invalid solution under official verify_denoising check.",
        }

    reward = 1.0 / max(mse, 1e-8)

    return {
        "ok": True,
        "mse": mse,
        "poisson": poisson,
        "reward": reward,
        "raw_score": mse,
        "metrics": metrics,
        "error": None,
    }


def evaluate_candidate(code: str, eval_mode: str):
    if eval_mode == "dummy":
        return evaluate_candidate_dummy(code)

    if eval_mode == "official":
        return evaluate_candidate_official(code)

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def make_child_state(step: int, code: str, eval_result: dict):
    """
    Create official DenoisingState child for PUCT.

    Official denoising minimizes MSE.
    PUCT maximizes value.
    Therefore:
        value = -mse
    """
    mse = eval_result.get("mse")
    poisson = eval_result.get("poisson")

    if mse is None:
        value = None
    else:
        value = -float(mse)

    return DenoisingState(
        timestep=step,
        construction=[],
        code=code,
        value=value,
        mse=mse,
        poisson=poisson,
        observation="",
    )


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_puct_sampler(output_dir: Path, rollouts_per_step: int):
    sampler_path = output_dir / "puct_sampler.json"

    return PUCTSampler(
        file_path=str(sampler_path),
        env_type=LocalDenoisingEnvForPUCT,
        problem_type="",
        max_buffer_size=1000,
        batch_size=rollouts_per_step,
        puct_c=1.0,
        topk_children=2,
    )


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
        default=int(os.environ.get("LOCAL_TTT_ROLLOUTS", "1")),
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

    print_section("Tiny Local TTT-Denoising with Official Prompt, Reward Check, and PUCT")
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

    print_section("Create PUCT Sampler")
    sampler = create_puct_sampler(
        output_dir=output_dir,
        rollouts_per_step=args.rollouts_per_step,
    )

    print("Initial PUCT stats:")
    print(json.dumps(sampler.get_sample_stats(), indent=2))

    print_section("Load Local LLM + LoRA")
    tokenizer, model = load_tokenizer_and_lora_model(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
    )
    optimizer = make_optimizer(model, learning_rate=args.learning_rate)

    history = []

    best_record = {
        "ok": False,
        "mse": None,
        "poisson": None,
        "reward": -1.0,
        "raw_score": None,
        "metrics": {},
        "code": None,
        "source": None,
    }

    for step in range(1, args.steps + 1):
        print_section(f"Step {step}/{args.steps}")

        parent_states = sampler.sample_states(args.rollouts_per_step)

        rollout_prompts = []
        rollout_responses = []
        rollout_rewards = []

        valid_child_states = []
        valid_parent_states = []

        for rollout_id, parent_state in enumerate(parent_states, start=1):
            print_section(f"Step {step} Rollout {rollout_id}/{args.rollouts_per_step}")

            print("Parent state:")
            print("  id:", parent_state.id)
            print("  timestep:", parent_state.timestep)
            print("  value:", parent_state.value)
            print("  mse:", getattr(parent_state, "mse", None))
            print("  poisson:", getattr(parent_state, "poisson", None))

            start_time = time.time()

            prompt = build_denoising_prompt(
                current_code=parent_state.code,
                history=history,
            )

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
                "parent_id": parent_state.id,
                "parent_timestep": parent_state.timestep,
                "parent_value": parent_state.value,
                "parent_mse": getattr(parent_state, "mse", None),
                "parent_poisson": getattr(parent_state, "poisson", None),
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
            }

            history.append(record)

            rollout_prompts.append(generation["rendered_prompt"])
            rollout_responses.append(generation["raw_text"])
            rollout_rewards.append(float(eval_result.get("reward", -1.0)))

            print("Evaluation result:")
            print(json.dumps(eval_result, indent=2))
            print(f"Elapsed seconds: {elapsed:.2f}")

            if record["ok"]:
                child_state = make_child_state(
                    step=step,
                    code=code,
                    eval_result=eval_result,
                )

                valid_child_states.append(child_state)
                valid_parent_states.append(parent_state)

                if record["reward"] > best_record["reward"]:
                    best_record = {
                        **record,
                        "source": f"step_{step}_rollout_{rollout_id}",
                    }
            else:
                sampler.record_failed_rollout(parent_state)

        print_section("Update PUCT Sampler")

        if valid_child_states:
            sampler.update_states(
                states=valid_child_states,
                parent_states=valid_parent_states,
                save=True,
                step=step,
            )
            print(f"Added {len(valid_child_states)} valid child states to PUCT.")
        else:
            sampler.flush(step=step)
            print("No valid child states. Flushed PUCT sampler only.")

        puct_stats = sampler.get_sample_stats()

        print("PUCT stats:")
        print(json.dumps(puct_stats, indent=2))

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
                "puct_stats": puct_stats,
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
        "raw_score": best_record.get("raw_score"),
        "metrics": best_record.get("metrics", {}),
        "source": best_record.get("source"),
    }

    print(json.dumps(summary, indent=2))

    save_json(output_dir / "summary.json", summary)

    if best_record.get("code"):
        best_code_path = output_dir / "best_magic_denoise.py"
        best_code_path.write_text(best_record["code"], encoding="utf-8")
        print("Saved best code to:", best_code_path)
    else:
        print("No valid best code was found.")


if __name__ == "__main__":
    main()
