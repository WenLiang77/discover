from __future__ import annotations

from epidemic_forecasting.tasks.base import EpidemicForecastTask
from epidemic_forecasting.tasks.covid19.task import (
    create_covid19_task,
)


def test_dataset(dataset: str) -> None:
    task = create_covid19_task(
        dataset=dataset,
        forecast_horizon=14,
        runtime_budget_seconds=20,
        random_state=0,
    )

    print(f"\nDataset: {dataset}")
    print("Protocol compatible:", isinstance(task, EpidemicForecastTask))
    print("Description:", task.describe())

    train_values, test_values, metadata = task.load_data()

    print("Train shape:", train_values.shape)
    print("Test shape:", test_values.shape)
    print("Locations:", metadata["number_of_locations"])

    result = task.evaluate_initial_code()

    print("Baseline valid:", result.ok)
    print("Baseline reward:", result.reward)
    print("Baseline metrics:", result.metrics)
    print("Baseline error:", result.error)


def main() -> None:
    test_dataset("uk")
    test_dataset("us")


if __name__ == "__main__":
    main()
