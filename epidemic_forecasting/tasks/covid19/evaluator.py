from __future__ import annotations

import ast
import builtins
import hashlib
import math
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from epidemic_forecasting.tasks.base import (
    EvaluationResult,
    TaskConfig,
    metric_to_reward,
)
from epidemic_forecasting.tasks.covid19.config import (
    make_covid19_config,
)
from epidemic_forecasting.tasks.covid19.data_loader import (
    load_covid19_data,
)


ALLOWED_IMPORT_ROOTS = {
    "collections",
    "functools",
    "itertools",
    "math",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statistics",
    "torch",
    "typing",
    "warnings",
}

ALLOWED_IMPORT_PREFIXES = {
    "statsmodels.tsa",
}


def _import_is_allowed(module_name: str) -> bool:
    """Allow ordinary scientific roots and selected statsmodels modules."""
    root = module_name.split(".", maxsplit=1)[0]

    if root in ALLOWED_IMPORT_ROOTS:
        return True

    return any(
        module_name == prefix
        or module_name.startswith(prefix + ".")
        for prefix in ALLOWED_IMPORT_PREFIXES
    )


BLOCKED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

BLOCKED_ATTRIBUTE_NAMES = {
    "__bases__",
    "__builtins__",
    "__class__",
    "__code__",
    "__dict__",
    "__func__",
    "__getattribute__",
    "__globals__",
    "__mro__",
    "__reduce__",
    "__reduce_ex__",
    "__self__",
    "__subclasses__",
    "dump",
    "dumps",
    "fromfile",
    "genfromtxt",
    "load",
    "loads",
    "loadtxt",
    "memmap",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_hdf",
    "read_html",
    "read_json",
    "read_parquet",
    "read_pickle",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_parquet",
    "to_pickle",
    "tofile",
}



def _safe_constant_expression(node: ast.AST) -> bool:
    """Return whether a top-level assignment contains only constants."""
    if isinstance(node, ast.Constant):
        return True

    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_safe_constant_expression(item) for item in node.elts)

    if isinstance(node, ast.Dict):
        return all(
            key is None or _safe_constant_expression(key)
            for key in node.keys
        ) and all(
            _safe_constant_expression(value)
            for value in node.values
        )

    if isinstance(node, ast.UnaryOp):
        return (
            isinstance(node.op, (ast.UAdd, ast.USub, ast.Not))
            and _safe_constant_expression(node.operand)
        )

    if isinstance(node, ast.BinOp):
        return (
            isinstance(
                node.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.Pow,
                ),
            )
            and _safe_constant_expression(node.left)
            and _safe_constant_expression(node.right)
        )

    return False


def validate_candidate_code(
    code: str,
    function_name: str = "covid_forecast",
) -> None:
    """
    Reject generated code that violates the task interface or obvious
    filesystem, network, subprocess, introspection, or dynamic-code rules.

    This is a defensive filter, not a complete security sandbox.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Candidate code is empty.")

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ValueError(
            f"Candidate code has invalid Python syntax: {error}"
        ) from error

    function_definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]

    if len(function_definitions) != 1:
        raise ValueError(
            f"Candidate code must define exactly one function named "
            f"{function_name!r}."
        )

    if isinstance(function_definitions[0], ast.AsyncFunctionDef):
        raise ValueError(
            f"{function_name!r} must be a normal synchronous function."
        )

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue

        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue

        if isinstance(node, ast.Assign):
            if not _safe_constant_expression(node.value):
                raise ValueError(
                    "Top-level assignments may contain only constants."
                )
            continue

        if isinstance(node, ast.AnnAssign):
            if (
                node.value is not None
                and not _safe_constant_expression(node.value)
            ):
                raise ValueError(
                    "Top-level annotated assignments may contain only "
                    "constants."
                )
            continue

        raise ValueError(
            "Candidate code may contain only imports, function definitions, "
            "constant assignments, and a module docstring at top level."
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _import_is_allowed(alias.name):
                    raise ValueError(
                        f"Importing {alias.name!r} is not allowed."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ValueError("Relative imports are not allowed.")

            if not _import_is_allowed(node.module):
                raise ValueError(
                    f"Importing from {node.module!r} is not allowed."
                )

        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global and nonlocal statements are not allowed.")

        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in BLOCKED_CALL_NAMES
            ):
                raise ValueError(
                    f"Calling {node.func.id!r} is not allowed."
                )

        elif isinstance(node, ast.Attribute):
            if (
                node.attr.startswith("__")
                or node.attr in BLOCKED_ATTRIBUTE_NAMES
            ):
                raise ValueError(
                    f"Accessing attribute {node.attr!r} is not allowed."
                )


def _controlled_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    """Import only libraries permitted by the forecasting prompt."""
    if level != 0:
        raise ImportError("Relative imports are not allowed.")

    if not _import_is_allowed(name):
        raise ImportError(f"Importing {name!r} is not allowed.")

    return builtins.__import__(
        name,
        globals_,
        locals_,
        fromlist,
        level,
    )


def _restricted_builtins() -> dict[str, Any]:
    """Return the small built-in namespace exposed to generated code."""
    allowed_names = {
        "ArithmeticError",
        "AssertionError",
        "Exception",
        "FloatingPointError",
        "IndexError",
        "KeyError",
        "MemoryError",
        "OverflowError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "pow",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }

    namespace = {
        name: getattr(builtins, name)
        for name in allowed_names
    }
    namespace["__import__"] = _controlled_import
    return namespace


def _candidate_worker(
    result_queue,
    code: str,
    function_name: str,
    train_values: np.ndarray,
    horizon: int,
    candidate_kwargs: dict[str, Any],
) -> None:
    """Execute one generated forecasting function in a child process."""
    try:
        namespace: dict[str, Any] = {
            "__builtins__": _restricted_builtins(),
            "__name__": "generated_epidemic_forecaster",
        }

        compiled = compile(
            code,
            filename="<generated_forecaster>",
            mode="exec",
        )
        exec(compiled, namespace, namespace)

        function = namespace.get(function_name)

        if not callable(function):
            raise TypeError(
                f"Generated code did not create callable "
                f"{function_name!r}."
            )

        local_train_values = np.asarray(
            train_values,
            dtype=np.float64,
        ).copy()

        started = time.perf_counter()
        output = function(
            local_train_values,
            int(horizon),
            **candidate_kwargs,
        )
        runtime_seconds = time.perf_counter() - started

        predictions = np.asarray(output, dtype=np.float64)

        result_queue.put(
            {
                "ok": True,
                "predictions": predictions,
                "runtime_seconds": runtime_seconds,
            }
        )

    except BaseException as error:
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=12),
            }
        )


def run_candidate_in_subprocess(
    code: str,
    function_name: str,
    train_values: np.ndarray,
    horizon: int,
    runtime_budget_seconds: float,
    candidate_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run generated code with a wall-clock timeout."""
    if runtime_budget_seconds <= 0:
        raise ValueError("runtime_budget_seconds must be positive.")

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)

    process = context.Process(
        target=_candidate_worker,
        args=(
            result_queue,
            code,
            function_name,
            train_values,
            horizon,
            candidate_kwargs,
        ),
    )

    process.start()
    process.join(timeout=float(runtime_budget_seconds))

    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        result_queue.close()

        return {
            "ok": False,
            "error": (
                "TimeoutError: candidate exceeded the runtime budget of "
                f"{runtime_budget_seconds:.3f} seconds."
            ),
            "timed_out": True,
        }

    try:
        payload = result_queue.get(timeout=2.0)
    except queue.Empty:
        payload = {
            "ok": False,
            "error": (
                "Candidate process exited without returning a result. "
                f"Exit code: {process.exitcode}."
            ),
        }
    finally:
        result_queue.close()
        result_queue.join_thread()

    return payload


def validate_predictions(
    predictions: np.ndarray,
    expected_shape: tuple[int, int],
    allow_negative: bool = False,
    negative_tolerance: float = 1e-10,
) -> np.ndarray:
    """Validate and return a dense floating-point forecast array."""
    array = np.asarray(predictions, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError(
            "Forecast must be a two-dimensional NumPy array, "
            f"but received shape {array.shape}."
        )

    if array.shape != expected_shape:
        raise ValueError(
            f"Forecast has shape {array.shape}; expected "
            f"{expected_shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Forecast contains NaN or infinite values."
        )

    if not allow_negative:
        minimum = float(array.min())

        if minimum < -negative_tolerance:
            raise ValueError(
                "Forecast contains negative values. "
                f"Minimum prediction: {minimum}."
            )

        array = np.maximum(array, 0.0)

    return np.ascontiguousarray(array, dtype=np.float64)


def calculate_smape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Symmetric mean absolute percentage error on a 0-200 scale."""
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


def calculate_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Mean absolute error."""
    return float(
        np.mean(
            np.abs(
                np.asarray(actual, dtype=np.float64)
                - np.asarray(predicted, dtype=np.float64)
            )
        )
    )


def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Root mean squared error."""
    errors = (
        np.asarray(actual, dtype=np.float64)
        - np.asarray(predicted, dtype=np.float64)
    )
    return float(np.sqrt(np.mean(np.square(errors))))


def calculate_mase(
    train_values: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonality: int = 1,
    epsilon: float = 1e-12,
) -> float:
    """
    Mean absolute scaled error, averaged across regions.

    The scaling error is calculated from a seasonal-naive forecast on the
    training period. EpiCastBench's current evaluator calls MASE without
    specifying a seasonal period, so the default is one time step.
    """
    train_values = np.asarray(train_values, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if seasonality < 1:
        raise ValueError("seasonality must be at least 1.")

    if train_values.shape[0] <= seasonality:
        raise ValueError(
            "Training data is too short to calculate MASE."
        )

    forecast_error_by_region = np.mean(
        np.abs(actual - predicted),
        axis=0,
    )
    naive_scale_by_region = np.mean(
        np.abs(
            train_values[seasonality:]
            - train_values[:-seasonality]
        ),
        axis=0,
    )

    region_scores = np.empty_like(
        forecast_error_by_region,
        dtype=np.float64,
    )

    stable = naive_scale_by_region > epsilon
    region_scores[stable] = (
        forecast_error_by_region[stable]
        / naive_scale_by_region[stable]
    )

    constant = ~stable
    region_scores[constant] = np.where(
        forecast_error_by_region[constant] <= epsilon,
        0.0,
        forecast_error_by_region[constant] / epsilon,
    )

    return float(np.mean(region_scores))


def calculate_metrics(
    train_values: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    mase_seasonality: int = 1,
) -> dict[str, float]:
    """Calculate the four EpiCastBench point-forecast metrics."""
    metrics = {
        "smape": calculate_smape(actual, predicted),
        "mae": calculate_mae(actual, predicted),
        "rmse": calculate_rmse(actual, predicted),
        "mase": calculate_mase(
            train_values,
            actual,
            predicted,
            seasonality=mase_seasonality,
        ),
    }

    for name, value in metrics.items():
        if not math.isfinite(value):
            raise ValueError(
                f"Metric {name!r} is not finite: {value}."
            )

    return metrics


def prediction_behavior_signature(
    predictions: np.ndarray,
    decimals: int = 6,
) -> str:
    """Hash rounded predictions to detect duplicate forecast behaviour."""
    rounded = np.round(
        np.asarray(predictions, dtype=np.float64),
        decimals=decimals,
    )
    digest = hashlib.sha256()
    digest.update(str(rounded.shape).encode("utf-8"))
    digest.update(rounded.tobytes(order="C"))
    return digest.hexdigest()


class Covid19Evaluator:
    """
    Load one held-out COVID-19 task and evaluate generated code against it.
    """

    def __init__(
        self,
        dataset: str | Path = "uk",
        forecast_horizon: int = 14,
        config: TaskConfig | None = None,
        runtime_budget_seconds: float | None = None,
        random_state: int = 0,
        mase_seasonality: int = 1,
    ) -> None:
        dataset_id = (
            Path(dataset).stem
            if isinstance(dataset, Path)
            else str(dataset)
        )

        self.config = config or make_covid19_config(
            forecast_horizon=forecast_horizon,
            dataset_id=dataset_id,
            frequency="daily",
        )

        if self.config.forecast_horizon != forecast_horizon:
            raise ValueError(
                "config.forecast_horizon must match forecast_horizon."
            )

        self.train_values, self.test_values, self.data_metadata = (
            load_covid19_data(
                dataset=dataset,
                forecast_horizon=forecast_horizon,
            )
        )

        configured_budget = float(
            self.config.metadata.get(
                "runtime_budget_seconds",
                400.0,
            )
        )

        self.runtime_budget_seconds = float(
            runtime_budget_seconds
            if runtime_budget_seconds is not None
            else configured_budget
        )
        self.random_state = int(random_state)
        self.mase_seasonality = int(mase_seasonality)

    def evaluate_code(self, code: str) -> EvaluationResult:
        """Validate, execute, score, and package one generated solution."""
        try:
            validate_candidate_code(
                code,
                function_name=self.config.function_name,
            )
        except Exception as error:
            return EvaluationResult(
                ok=False,
                error=f"Static validation failed: {error}",
                metadata={
                    "dataset_id": self.data_metadata["dataset_id"],
                    "stage": "static_validation",
                },
            )

        candidate_kwargs = {
            "budget_s": self.runtime_budget_seconds,
            "random_state": self.random_state,
            "frequency": self.data_metadata["frequency"],
            "time_index": list(
                self.data_metadata.get("train_dates", [])
            ),
        }

        started = time.perf_counter()
        payload = run_candidate_in_subprocess(
            code=code,
            function_name=self.config.function_name,
            train_values=self.train_values,
            horizon=self.config.forecast_horizon,
            runtime_budget_seconds=self.runtime_budget_seconds,
            candidate_kwargs=candidate_kwargs,
        )
        total_runtime_seconds = time.perf_counter() - started

        if not payload.get("ok"):
            return EvaluationResult(
                ok=False,
                error=str(
                    payload.get(
                        "error",
                        "Candidate execution failed.",
                    )
                ),
                metadata={
                    "dataset_id": self.data_metadata["dataset_id"],
                    "stage": "execution",
                    "runtime_seconds": total_runtime_seconds,
                    "traceback": payload.get("traceback"),
                    "timed_out": bool(
                        payload.get("timed_out", False)
                    ),
                },
            )

        try:
            predictions = validate_predictions(
                payload["predictions"],
                expected_shape=self.test_values.shape,
                allow_negative=self.config.allow_negative_predictions,
            )

            metrics = calculate_metrics(
                train_values=self.train_values,
                actual=self.test_values,
                predicted=predictions,
                mase_seasonality=self.mase_seasonality,
            )

            primary_value = metrics[self.config.primary_metric]
            reward = metric_to_reward(
                primary_value,
                self.config.primary_metric_direction,
            )

            signature = prediction_behavior_signature(predictions)

        except Exception as error:
            return EvaluationResult(
                ok=False,
                error=f"Evaluation failed: {error}",
                metadata={
                    "dataset_id": self.data_metadata["dataset_id"],
                    "stage": "prediction_validation_or_metrics",
                    "runtime_seconds": total_runtime_seconds,
                },
            )

        return EvaluationResult(
            ok=True,
            reward=float(reward),
            metrics=metrics,
            predictions=predictions,
            behavior_signature=signature,
            metadata={
                "dataset_id": self.data_metadata["dataset_id"],
                "forecast_horizon": self.config.forecast_horizon,
                "number_of_locations": (
                    self.data_metadata["number_of_locations"]
                ),
                "candidate_runtime_seconds": float(
                    payload.get("runtime_seconds", 0.0)
                ),
                "total_runtime_seconds": total_runtime_seconds,
                "primary_metric": self.config.primary_metric,
                "primary_metric_direction": (
                    self.config.primary_metric_direction
                ),
            },
        )

def evaluate_covid19_code(
    code: str,
    dataset: str | Path = "uk",
    forecast_horizon: int = 14,
    runtime_budget_seconds: float | None = None,
    random_state: int = 0,
) -> EvaluationResult:
    """Convenience function for one complete candidate evaluation."""
    evaluator = Covid19Evaluator(
        dataset=dataset,
        forecast_horizon=forecast_horizon,
        runtime_budget_seconds=runtime_budget_seconds,
        random_state=random_state,
    )
    return evaluator.evaluate_code(code)


if __name__ == "__main__":
    print(
        "Import Covid19Evaluator from a guarded main script to run "
        "candidate evaluations."
    )
