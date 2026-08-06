from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def model_result(payload: dict[str, Any]) -> dict[str, Any]:
    return (
        payload.get("best_model_result")
        or payload.get("best_generated_attempt")
        or payload.get("best_generated_state")
        or {}
    )


def summary_to_row(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    task = payload.get("task", {}) or {}
    best = model_result(payload)
    best_metrics = best.get("metrics", {}) or {}

    return {
        "summary_path": str(path),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "runner": payload.get("runner"),
        "task_id": task.get("task_id"),
        "dataset_id": task.get("dataset_id"),
        "forecast_horizon": task.get("forecast_horizon"),
        "valid_generated_count": payload.get("valid_generated_count"),
        "invalid_generated_count": payload.get("invalid_generated_count"),
        "duplicate_behavior_count": payload.get(
            "duplicate_behavior_count"
        ),
        "best_attempt": best.get("attempt"),
        "best_step": best.get("step"),
        "best_parent_number": best.get("parent_number"),
        "best_rollout_number": best.get("rollout_number"),
        "best_candidate_path": best.get("candidate_path"),
        "best_reward": first_present(best, "raw_reward", "reward"),
        "best_adjusted_reward": best.get("adjusted_reward"),
        "best_smape": best_metrics.get("smape"),
        "best_mae": best_metrics.get("mae"),
        "best_rmse": best_metrics.get("rmse"),
        "best_mase": best_metrics.get("mase"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect model-generated epidemic forecasting results "
            "into one comparison CSV."
        )
    )
    parser.add_argument(
        "--results-root",
        default="epidemic_forecasting/results",
    )
    parser.add_argument(
        "--output",
        default=(
            "epidemic_forecasting/results/"
            "experiment_comparison.csv"
        ),
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    summary_paths = sorted(results_root.rglob("summary.json"))
    rows = [summary_to_row(path) for path in summary_paths]

    if not rows:
        raise SystemExit(
            f"No summary.json files found under {results_root}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} experiment row(s) to:")
    print(output_path)


if __name__ == "__main__":
    main()
