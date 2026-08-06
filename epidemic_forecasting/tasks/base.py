from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


MetricDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class TaskConfig:
    """
    Configuration shared by one epidemic forecasting task.

    A task normally represents one disease, dataset and forecasting setup.
    """

    task_id: str
    disease_name: str
    function_name: str
    prompt_path: Path
    forecast_horizon: int

    metric_names: tuple[str, ...]
    primary_metric: str
    primary_metric_direction: MetricDirection = "minimize"

    allow_negative_predictions: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_path",
            Path(self.prompt_path),
        )

        if not self.task_id.strip():
            raise ValueError("task_id must not be empty.")

        if not self.disease_name.strip():
            raise ValueError("disease_name must not be empty.")

        if not self.function_name.strip():
            raise ValueError("function_name must not be empty.")

        if self.forecast_horizon < 1:
            raise ValueError(
                "forecast_horizon must be at least 1."
            )

        if not self.metric_names:
            raise ValueError(
                "metric_names must contain at least one metric."
            )

        if self.primary_metric not in self.metric_names:
            raise ValueError(
                "primary_metric must be included in metric_names."
            )

        if self.primary_metric_direction not in {
            "minimize",
            "maximize",
        }:
            raise ValueError(
                "primary_metric_direction must be either "
                "'minimize' or 'maximize'."
            )


@dataclass
class EvaluationResult:
    """
    Standard result returned after evaluating generated forecasting code.
    """

    ok: bool
    reward: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray | None = None
    error: str | None = None
    behavior_signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_history_item(self) -> dict[str, Any]:
        """
        Convert this result into the format expected by generation.py.
        """
        return {
            "ok": self.ok,
            "reward": self.reward,
            "metrics": dict(self.metrics),
            "error": self.error,
            "behavior_signature": self.behavior_signature,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class EpidemicForecastTask(Protocol):
    """
    Interface that every disease-specific task must implement.
    """

    config: TaskConfig

    def load_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Return:

        train_values:
            Historical observations available to the forecasting method.

        test_values:
            Held-out observations used only during evaluation.

        metadata:
            Additional information such as dates, locations and feature names.
        """
        ...

    def evaluate_code(
        self,
        code: str,
    ) -> EvaluationResult:
        """
        Execute and evaluate one generated forecasting implementation.
        """
        ...


def metric_to_reward(
    metric_value: float,
    direction: MetricDirection,
) -> float:
    """
    Convert a task metric into a reward where larger is always better.

    For an error metric such as SMAPE:
        direction = "minimize"
        reward = -SMAPE

    For a score that is already larger-is-better:
        direction = "maximize"
        reward = score
    """
    value = float(metric_value)

    if not np.isfinite(value):
        raise ValueError("Metric value must be finite.")

    if direction == "minimize":
        return -value

    if direction == "maximize":
        return value

    raise ValueError(
        f"Unsupported metric direction: {direction}"
    )


if __name__ == "__main__":
    print(
        "This module defines the common epidemic task interface."
    )
