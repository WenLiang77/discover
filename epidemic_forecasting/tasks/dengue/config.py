from __future__ import annotations

import re
from pathlib import Path

from epidemic_forecasting.tasks.base import TaskConfig


TASK_DIRECTORY = Path(__file__).resolve().parent
PROMPT_PATH = TASK_DIRECTORY / "prompt.txt"

FUNCTION_NAME = "dengue_forecast"

SUPPORTED_FORECAST_HORIZONS = (8,)
DEFAULT_FORECAST_HORIZON = 8

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
        "Dengue Colombia" -> "dengue_colombia"
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = value.strip("_")

    if not value:
        return "unspecified"

    return value


def make_dengue_config(
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    dataset_id: str = "unspecified",
    frequency: str = "weekly",
) -> TaskConfig:
    """
    Create a configuration for one dengue forecasting experiment.

    Parameters
    ----------
    forecast_horizon:
        Number of future weekly time steps to predict.

    dataset_id:
        Identifier of the dataset used by the experiment, for example
        "dengue_colombia" or "dengue_panama".

    frequency:
        Time-series frequency. The selected datasets use weekly data.
    """
    if forecast_horizon not in SUPPORTED_FORECAST_HORIZONS:
        raise ValueError(
            "Unsupported dengue forecast horizon. "
            f"Expected one of {SUPPORTED_FORECAST_HORIZONS}, "
            f"but received {forecast_horizon}."
        )

    clean_dataset_id = _normalise_identifier(dataset_id)
    clean_frequency = frequency.strip().lower()

    if not clean_frequency:
        raise ValueError("frequency must not be empty.")

    if clean_frequency != "weekly":
        raise ValueError(
            "The current dengue experiments expect weekly data."
        )

    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(
            f"Dengue prompt file not found: {PROMPT_PATH}"
        )

    return TaskConfig(
        task_id=(
            f"dengue_{clean_dataset_id}_"
            f"{forecast_horizon}weeks"
        ),
        disease_name="Dengue",
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


DENGUE_CONFIG = make_dengue_config()


if __name__ == "__main__":
    print("Dengue task configuration:")
    print(DENGUE_CONFIG)
