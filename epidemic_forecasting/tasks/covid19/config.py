from __future__ import annotations

import re
from pathlib import Path

from epidemic_forecasting.tasks.base import TaskConfig


TASK_DIRECTORY = Path(__file__).resolve().parent
PROMPT_PATH = TASK_DIRECTORY / "prompt.txt"

FUNCTION_NAME = "covid_forecast"

SUPPORTED_FORECAST_HORIZONS = (7, 14, 30)
DEFAULT_FORECAST_HORIZON = 14

METRIC_NAMES = (
    "smape",
    "mae",
    "rmse",
    "mase",
)

PRIMARY_METRIC = "smape"

RUNTIME_BUDGET_SECONDS = 400


def _normalise_identifier(value: str) -> str:
    """
    Convert a human-readable dataset name into a safe task identifier.

    Example:
        "COVID UK" -> "covid_uk"
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = value.strip("_")

    if not value:
        return "unspecified"

    return value


def make_covid19_config(
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    dataset_id: str = "unspecified",
    frequency: str = "daily",
) -> TaskConfig:
    """
    Create a configuration for one COVID-19 forecasting experiment.

    Parameters
    ----------
    forecast_horizon:
        Number of future time steps to predict.

    dataset_id:
        Identifier of the dataset used by the experiment, for example
        "covid_uk" or "covid_us".

    frequency:
        Time-series frequency, normally "daily" for the current prompt.
    """
    if forecast_horizon not in SUPPORTED_FORECAST_HORIZONS:
        raise ValueError(
            "Unsupported COVID-19 forecast horizon. "
            f"Expected one of {SUPPORTED_FORECAST_HORIZONS}, "
            f"but received {forecast_horizon}."
        )

    clean_dataset_id = _normalise_identifier(dataset_id)
    clean_frequency = frequency.strip().lower()

    if not clean_frequency:
        raise ValueError("frequency must not be empty.")

    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(
            f"COVID-19 prompt file not found: {PROMPT_PATH}"
        )

    return TaskConfig(
        task_id=(
            f"covid19_{clean_dataset_id}_"
            f"{forecast_horizon}days"
        ),
        disease_name="COVID-19",
        function_name=FUNCTION_NAME,
        prompt_path=PROMPT_PATH,
        forecast_horizon=forecast_horizon,
        metric_names=METRIC_NAMES,
        primary_metric=PRIMARY_METRIC,
        primary_metric_direction="minimize",
        allow_negative_predictions=False,
        metadata={
            "dataset_id": clean_dataset_id,
            "frequency": clean_frequency,
            "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
            "supported_forecast_horizons": (
                SUPPORTED_FORECAST_HORIZONS
            ),
            "task_version": 1,
        },
    )


COVID19_CONFIG = make_covid19_config()


if __name__ == "__main__":
    print("COVID-19 task configuration:")
    print(COVID19_CONFIG)
