from __future__ import annotations

from epidemic_forecasting.tasks.covid19.evaluator import (
    Covid19Evaluator,
)


def print_result(dataset: str) -> None:
    evaluator = Covid19Evaluator(
        dataset=dataset,
        forecast_horizon=14,
        runtime_budget_seconds=20,
        random_state=0,
    )

    result = evaluator.evaluate_naive_baseline()

    print(f"\nDataset: {dataset}")
    print(f"Valid: {result.ok}")
    print(f"Reward: {result.reward}")
    print(f"Metrics: {result.metrics}")
    print(f"Error: {result.error}")

    if result.predictions is not None:
        print(f"Prediction shape: {result.predictions.shape}")

    print(f"Metadata: {result.metadata}")


def main() -> None:
    print_result("uk")
    print_result("us")


if __name__ == "__main__":
    main()
