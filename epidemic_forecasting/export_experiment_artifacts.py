from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from epidemic_forecasting.tasks.covid19.task import create_covid19_task


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denominator = np.abs(actual) + np.abs(predicted)
    numerator = 2.0 * np.abs(actual - predicted)
    terms = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )
    return float(100.0 * np.mean(terms))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(actual, dtype=np.float64)
                - np.asarray(predicted, dtype=np.float64)
            )
        )
    )


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    error = (
        np.asarray(actual, dtype=np.float64)
        - np.asarray(predicted, dtype=np.float64)
    )
    return float(np.sqrt(np.mean(np.square(error))))


def model_result(summary: dict[str, Any]) -> dict[str, Any]:
    return (
        summary.get("best_model_result")
        or summary.get("best_generated_attempt")
        or summary.get("best_generated_state")
        or {}
    )


def collect_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    runner = summary.get("runner")
    records: list[dict[str, Any]] = []

    def reward_fields(
        evaluation: dict[str, Any],
        selection_reward: Any = None,
        adjusted_reward: Any = None,
    ) -> dict[str, Any]:
        metadata = evaluation.get("metadata", {}) or {}

        evaluator_reward = evaluation.get("reward")

        if selection_reward is None:
            selection_reward = evaluator_reward

        if adjusted_reward is None:
            adjusted_reward = selection_reward

        return {
            # Reward before the flat-forecast penalty.
            "base_reward": metadata.get("base_reward"),

            # Penalty introduced by the flat-forecast detector.
            "flat_forecast_penalty": metadata.get(
                "flat_forecast_penalty"
            ),
            "flat_series_count": metadata.get(
                "flat_series_count"
            ),
            "flat_series_indices": metadata.get(
                "flat_series_indices"
            ),
            "flat_series_names": metadata.get(
                "flat_series_names"
            ),

            # evaluator_reward =
            # base_reward - flat_forecast_penalty
            "evaluator_reward": evaluator_reward,

            # Reward used when selecting the generated candidate.
            "selection_reward": selection_reward,

            # TTT may additionally apply duplicate penalties.
            "adjusted_reward": adjusted_reward,

            # Keep the old field for backwards compatibility.
            "reward": selection_reward,
        }

    if runner == "qwen_only_baseline":
        for item in summary.get("generated_attempts", []):
            evaluation = item.get("evaluation", {}) or {}
            metrics = evaluation.get("metrics", {}) or {}

            records.append(
                {
                    "source": f"Attempt {item.get('attempt')}",
                    "attempt": item.get("attempt"),
                    "step": None,
                    "parent_number": None,
                    "rollout_number": None,
                    "ok": bool(evaluation.get("ok")),
                    "duplicate": bool(
                        item.get(
                            "duplicate_valid_behavior",
                            False,
                        )
                    ),
                    "smape": metrics.get("smape"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "mase": metrics.get("mase"),
                    "behavior_signature": evaluation.get(
                        "behavior_signature"
                    ),
                    **reward_fields(evaluation),
                }
            )

    elif runner == "local_ttt":
        for step_payload in summary.get("steps", []):
            for item in step_payload.get("rollouts", []):
                evaluation = item.get("evaluation", {}) or {}
                metrics = evaluation.get("metrics", {}) or {}

                step = item.get("step")
                rollout = item.get("rollout_number")
                parent = item.get("parent_number")

                records.append(
                    {
                        "source": (
                            f"Step {step}, Rollout {rollout}"
                            if parent in (None, 1)
                            else (
                                f"Step {step}, Parent {parent}, "
                                f"Rollout {rollout}"
                            )
                        ),
                        "attempt": None,
                        "step": step,
                        "parent_number": parent,
                        "rollout_number": rollout,
                        "ok": bool(evaluation.get("ok")),
                        "duplicate": bool(
                            item.get(
                                "duplicate_valid_behavior",
                                False,
                            )
                        ),
                        "smape": metrics.get("smape"),
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "mase": metrics.get("mase"),
                        "behavior_signature": evaluation.get(
                            "behavior_signature"
                        ),
                        **reward_fields(
                            evaluation,
                            selection_reward=item.get(
                                "raw_reward"
                            ),
                            adjusted_reward=item.get(
                                "adjusted_reward"
                            ),
                        ),
                    }
                )

    else:
        raise ValueError(
            f"Unsupported runner: {runner!r}"
        )

    return records



def save_top10_and_aggregate(
    run_dir: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    valid = [
        record
        for record in records
        if record["ok"] and record["smape"] is not None
    ]

    distinct_valid = [
        record
        for record in valid
        if not record["duplicate"]
    ]

    by_smape = sorted(
        distinct_valid,
        key=lambda record: float(record["smape"]),
    )

    by_reward = sorted(
        [
            record
            for record in distinct_valid
            if record.get("selection_reward") is not None
        ],
        key=lambda record: float(
            record["selection_reward"]
        ),
        reverse=True,
    )

    fieldnames = [
        "rank",
        "source",
        "attempt",
        "step",
        "parent_number",
        "rollout_number",
        "smape",
        "mae",
        "rmse",
        "mase",
        "base_reward",
        "flat_series_count",
        "flat_forecast_penalty",
        "selection_reward",
        "adjusted_reward",
    ]

    def write_ranking(
        output_path: Path,
        ranked_records: list[dict[str, Any]],
    ) -> None:
        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for rank, record in enumerate(
                ranked_records[:10],
                start=1,
            ):
                writer.writerow(
                    {
                        "rank": rank,
                        **{
                            name: record.get(name)
                            for name in fieldnames
                            if name != "rank"
                        },
                    }
                )

    top10_smape_path = (
        run_dir / "top10_by_smape.csv"
    )
    top10_reward_path = (
        run_dir / "top10_by_reward.csv"
    )

    write_ranking(
        top10_smape_path,
        by_smape,
    )
    write_ranking(
        top10_reward_path,
        by_reward,
    )

    # Keep the old filename so existing README/scripts
    # continue to work. It remains the raw-SMAPE ranking.
    legacy_path = (
        run_dir / "top10_distinct_results.csv"
    )
    shutil.copyfile(
        top10_smape_path,
        legacy_path,
    )

    all_smapes = np.asarray(
        [
            float(record["smape"])
            for record in valid
        ],
        dtype=np.float64,
    )

    distinct_smapes = np.asarray(
        [
            float(record["smape"])
            for record in distinct_valid
        ],
        dtype=np.float64,
    )

    best_smape_record = (
        by_smape[0]
        if by_smape
        else None
    )

    best_reward_record = (
        by_reward[0]
        if by_reward
        else None
    )

    aggregate = {
        "runner": summary.get("runner"),
        "task": summary.get("task"),

        "total_generated_count": len(records),
        "valid_generated_count": len(valid),
        "invalid_generated_count": (
            len(records) - len(valid)
        ),
        "duplicate_behavior_count": sum(
            1
            for record in valid
            if record["duplicate"]
        ),
        "distinct_valid_count": len(
            distinct_valid
        ),

        # Raw benchmark statistics.
        "best_smape": (
            float(np.min(all_smapes))
            if all_smapes.size
            else None
        ),
        "best_smape_source": (
            best_smape_record.get("source")
            if best_smape_record
            else None
        ),
        "median_smape_all_valid": (
            float(np.median(all_smapes))
            if all_smapes.size
            else None
        ),
        "median_smape_distinct_valid": (
            float(np.median(distinct_smapes))
            if distinct_smapes.size
            else None
        ),

        # Flat-forecast statistics.
        "flat_flagged_valid_count": sum(
            1
            for record in valid
            if (
                record.get("flat_series_count")
                is not None
                and int(
                    record["flat_series_count"]
                ) > 0
            )
        ),
        "flat_flagged_distinct_valid_count": sum(
            1
            for record in distinct_valid
            if (
                record.get("flat_series_count")
                is not None
                and int(
                    record["flat_series_count"]
                ) > 0
            )
        ),

        # Candidate preferred by the new reward.
        "best_selection_reward": (
            float(
                best_reward_record[
                    "selection_reward"
                ]
            )
            if best_reward_record
            else None
        ),
        "best_reward_source": (
            best_reward_record.get("source")
            if best_reward_record
            else None
        ),
        "best_reward_smape": (
            float(
                best_reward_record["smape"]
            )
            if best_reward_record
            else None
        ),
        "best_reward_base_reward": (
            best_reward_record.get(
                "base_reward"
            )
            if best_reward_record
            else None
        ),
        "best_reward_flat_series_count": (
            best_reward_record.get(
                "flat_series_count"
            )
            if best_reward_record
            else None
        ),
        "best_reward_flat_forecast_penalty": (
            best_reward_record.get(
                "flat_forecast_penalty"
            )
            if best_reward_record
            else None
        ),

        "ranking_files": {
            "by_smape": top10_smape_path.name,
            "by_reward": top10_reward_path.name,
            "legacy_by_smape": legacy_path.name,
        },
    }

    write_json(
        run_dir / "aggregate_results.json",
        aggregate,
    )



def save_best_forecast(
    run_dir: Path,
    summary: dict[str, Any],
    dataset: str,
    forecast_horizon: int,
) -> None:
    best = model_result(summary)
    if not best:
        raise RuntimeError(
            "No valid best model-generated result was found."
        )

    candidate_path_text = best.get("candidate_path")
    if not candidate_path_text:
        raise RuntimeError("Best result has no candidate_path.")

    candidate_path = Path(candidate_path_text)
    predictions_path = candidate_path.with_name("predictions.npy")

    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"Best candidate file not found: {candidate_path}"
        )
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Best prediction file not found: {predictions_path}"
        )

    copied_candidate = run_dir / "best_generated_candidate.py"
    shutil.copyfile(candidate_path, copied_candidate)

    predictions = np.asarray(
        np.load(predictions_path),
        dtype=np.float64,
    )

    task = create_covid19_task(
        dataset=dataset,
        forecast_horizon=forecast_horizon,
    )
    _, actual, metadata = task.load_data()

    if predictions.shape != actual.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match "
            f"actual shape {actual.shape}."
        )

    np.save(run_dir / "best_predictions.npy", predictions)
    np.save(run_dir / "actual_test_values.npy", actual)

    dates = metadata.get("test_dates", [])
    locations = metadata.get("locations", [])

    comparison_path = run_dir / "best_forecast_vs_actual.csv"
    with comparison_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "day",
            "date",
            "series_index",
            "series_name",
            "predicted",
            "actual",
            "absolute_error",
            "smape_component",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for day_index in range(actual.shape[0]):
            for series_index in range(actual.shape[1]):
                predicted_value = float(
                    predictions[day_index, series_index]
                )
                actual_value = float(actual[day_index, series_index])
                denominator = (
                    abs(predicted_value) + abs(actual_value)
                )
                smape_component = (
                    0.0
                    if denominator == 0.0
                    else (
                        200.0
                        * abs(actual_value - predicted_value)
                        / denominator
                    )
                )
                writer.writerow(
                    {
                        "day": day_index + 1,
                        "date": (
                            dates[day_index]
                            if day_index < len(dates)
                            else ""
                        ),
                        "series_index": series_index,
                        "series_name": (
                            locations[series_index]
                            if series_index < len(locations)
                            else f"series_{series_index}"
                        ),
                        "predicted": predicted_value,
                        "actual": actual_value,
                        "absolute_error": abs(
                            actual_value - predicted_value
                        ),
                        "smape_component": smape_component,
                    }
                )

    series_metrics_path = run_dir / "best_series_metrics.csv"
    with series_metrics_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "series_index",
            "series_name",
            "smape",
            "mae",
            "rmse",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for series_index in range(actual.shape[1]):
            actual_series = actual[:, series_index]
            predicted_series = predictions[:, series_index]
            writer.writerow(
                {
                    "series_index": series_index,
                    "series_name": (
                        locations[series_index]
                        if series_index < len(locations)
                        else f"series_{series_index}"
                    ),
                    "smape": _smape(
                        actual_series,
                        predicted_series,
                    ),
                    "mae": _mae(
                        actual_series,
                        predicted_series,
                    ),
                    "rmse": _rmse(
                        actual_series,
                        predicted_series,
                    ),
                }
            )

    if summary.get("runner") == "qwen_only_baseline":
        source = {
            "attempt": best.get("attempt"),
        }
    else:
        source = {
            "step": best.get("step"),
            "parent_number": best.get("parent_number"),
            "rollout_number": best.get("rollout_number"),
        }

    result_payload_path = candidate_path.with_name(
        "result.json"
    )

    result_payload = (
        load_json(result_payload_path)
        if result_payload_path.is_file()
        else {}
    )

    evaluation_payload = (
        result_payload.get("evaluation", {})
        or {}
    )
    evaluation_metadata = (
        evaluation_payload.get("metadata", {})
        or {}
    )

    selection_reward = best.get(
        "raw_reward",
        best.get(
            "reward",
            evaluation_payload.get("reward"),
        ),
    )

    adjusted_reward = result_payload.get(
        "adjusted_reward",
        best.get("adjusted_reward"),
    )

    best_result = {
        "runner": summary.get("runner"),
        "dataset": dataset,
        "forecast_horizon": forecast_horizon,
        "source": source,

        "reward": selection_reward,
        "selection_reward": selection_reward,
        "adjusted_reward": adjusted_reward,

        "base_reward": evaluation_metadata.get(
            "base_reward"
        ),
        "flat_forecast_penalty": (
            evaluation_metadata.get(
                "flat_forecast_penalty"
            )
        ),
        "flat_penalty_per_series": (
            evaluation_metadata.get(
                "flat_penalty_per_series"
            )
        ),
        "flat_series_count": (
            evaluation_metadata.get(
                "flat_series_count"
            )
        ),
        "flat_series_indices": (
            evaluation_metadata.get(
                "flat_series_indices"
            )
        ),
        "flat_series_names": (
            evaluation_metadata.get(
                "flat_series_names"
            )
        ),
        "flat_forecast_analysis": (
            evaluation_metadata.get(
                "flat_forecast_analysis"
            )
        ),

        "metrics": best.get(
            "metrics",
            evaluation_payload.get(
                "metrics",
                {},
            ),
        ),
        "files": {
            "candidate": copied_candidate.name,
            "forecast_vs_actual": comparison_path.name,
            "series_metrics": series_metrics_path.name,
            "predictions_npy": "best_predictions.npy",
            "actual_npy": "actual_test_values.npy",
        },
    }
    write_json(run_dir / "best_result.json", best_result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export aggregate, top-10, best-code and best-forecast "
            "artifacts for a completed COVID-19 experiment."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--dataset",
        choices=("uk", "us"),
        required=True,
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        choices=(7, 14, 30),
        required=True,
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.json"

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"summary.json not found: {summary_path}"
        )

    summary = load_json(summary_path)
    records = collect_records(summary)

    save_top10_and_aggregate(
        run_dir=run_dir,
        summary=summary,
        records=records,
    )
    save_best_forecast(
        run_dir=run_dir,
        summary=summary,
        dataset=args.dataset,
        forecast_horizon=args.forecast_horizon,
    )

    print("Exported experiment artifacts to:")
    print(run_dir)


if __name__ == "__main__":
    main()
