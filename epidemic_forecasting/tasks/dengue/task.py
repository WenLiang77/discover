from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from epidemic_forecasting.tasks.base import (
    EpidemicForecastTask,
    EvaluationResult,
    TaskConfig,
)

from epidemic_forecasting.tasks.dengue.config import (
    make_dengue_config,
)

from epidemic_forecasting.tasks.dengue.evaluator import (
    DengueEvaluator,
)


class DengueTask:
    """
    Complete dengue forecasting task used by baseline and TTT runners.

    This class joins together:
    - the task configuration;
    - the Colombia or Panama EpiCastBench data;
    - generated-code evaluation.
    """

    def __init__(
        self,
        dataset: str | Path = "colombia",
        forecast_horizon: int = 8,
        runtime_budget_seconds: float | None = None,
        random_state: int = 0,
        mase_seasonality: int = 1,
    ) -> None:

        self.dataset = dataset
        self.forecast_horizon = int(forecast_horizon)
        self.random_state = int(random_state)
        self.mase_seasonality = int(mase_seasonality)

        dataset_id = self._dataset_id(dataset)

        self.config: TaskConfig = make_dengue_config(
            forecast_horizon=self.forecast_horizon,
            dataset_id=dataset_id,
            frequency="weekly",
        )

        self.evaluator = DengueEvaluator(
            dataset=self.dataset,
            forecast_horizon=self.forecast_horizon,
            config=self.config,
            runtime_budget_seconds=runtime_budget_seconds,
            random_state=self.random_state,
            mase_seasonality=self.mase_seasonality,
        )

    @staticmethod
    def _dataset_id(
        dataset: str | Path,
    ) -> str:
        """
        Return a stable identifier for either an alias or CSV path.
        """

        if isinstance(dataset, Path):
            return dataset.stem

        value = str(dataset).strip()

        if not value:
            raise ValueError(
                "dataset must not be empty."
            )

        possible_path = Path(value)

        if possible_path.suffix.lower() == ".csv":
            return possible_path.stem

        return value

    def load_data(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        """
        Return defensive copies of the train/test arrays
        and task metadata.

        The generated forecasting function receives only
        train_values.

        test_values remain inside the evaluator and are used
        only for scoring.
        """

        return (
            self.evaluator.train_values.copy(),
            self.evaluator.test_values.copy(),
            dict(self.evaluator.data_metadata),
        )

    def evaluate_code(
        self,
        code: str,
    ) -> EvaluationResult:
        """
        Evaluate one generated dengue_forecast implementation.
        """

        return self.evaluator.evaluate_code(code)

    def describe(
        self,
    ) -> dict[str, Any]:
        """
        Return a concise JSON-friendly description
        of this task instance.
        """

        metadata = self.evaluator.data_metadata

        return {
            "task_id": self.config.task_id,
            "disease_name": self.config.disease_name,
            "dataset_id": metadata["dataset_id"],
            "function_name": self.config.function_name,
            "prompt_path": str(
                self.config.prompt_path
            ),
            "forecast_horizon": (
                self.config.forecast_horizon
            ),
            "frequency": metadata["frequency"],
            "number_of_locations": (
                metadata["number_of_locations"]
            ),
            "training_observations": (
                metadata["training_observations"]
            ),
            "test_observations": (
                self.config.forecast_horizon
            ),
            "primary_metric": (
                self.config.primary_metric
            ),
            "primary_metric_direction": (
                self.config.primary_metric_direction
            ),
            "runtime_budget_seconds": (
                self.evaluator.runtime_budget_seconds
            ),
        }


def create_dengue_task(
    dataset: str | Path = "colombia",
    forecast_horizon: int = 8,
    runtime_budget_seconds: float | None = None,
    random_state: int = 0,
    mase_seasonality: int = 1,
) -> DengueTask:
    """
    Convenience factory used by command-line experiment runners.
    """

    return DengueTask(
        dataset=dataset,
        forecast_horizon=forecast_horizon,
        runtime_budget_seconds=runtime_budget_seconds,
        random_state=random_state,
        mase_seasonality=mase_seasonality,
    )


def _check_protocol_compatibility() -> None:
    """
    Internal development check for the runtime-checkable
    task protocol.
    """

    task = create_dengue_task(
        dataset="colombia",
        forecast_horizon=8,
        runtime_budget_seconds=20,
    )

    if not isinstance(
        task,
        EpidemicForecastTask,
    ):
        raise TypeError(
            "DengueTask does not satisfy "
            "EpidemicForecastTask."
        )


if __name__ == "__main__":
    print(
        "Import create_dengue_task from a guarded "
        "runner script to construct this task."
    )
