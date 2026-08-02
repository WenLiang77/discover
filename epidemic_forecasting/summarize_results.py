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


def summary_to_row(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    task = payload.get("task", {}) or {}
    initial = payload.get("initial_baseline", {}) or {}
    initial_metrics = initial.get("metrics", {}) or {}
    best = payload.get("best_overall", {}) or {}
    best_metrics = best.get("metrics", {}) or {}

    return {
        "summary_path": str(path),
        "status": payload.get("status"),
        "runner": payload.get("runner"),
        "task_id": task.get("task_id"),
        "dataset_id": task.get("dataset_id"),
        "forecast_horizon": task.get("forecast_horizon"),
        "initial_reward": initial.get("reward"),
        "initial_smape": initial_metrics.get("smape"),
        "initial_mae": initial_metrics.get("mae"),
        "initial_rmse": initial_metrics.get("rmse"),
        "initial_mase": initial_metrics.get("mase"),
        "valid_generated_count": payload.get("valid_generated_count"),
        "invalid_generated_count": payload.get("invalid_generated_count"),
        "duplicate_behavior_count": payload.get(
            "duplicate_behavior_count"
        ),
        "best_source": best.get("source"),
        "best_reward": first_present(best, "raw_reward", "reward"),
        "best_smape": best_metrics.get("smape"),
        "best_mae": best_metrics.get("mae"),
        "best_rmse": best_metrics.get("rmse"),
        "best_mase": best_metrics.get("mase"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect epidemic forecasting summary.json files "
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
