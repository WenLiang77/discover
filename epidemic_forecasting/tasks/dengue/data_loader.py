from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPIDEMIC_DIRECTORY = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = EPIDEMIC_DIRECTORY / "data" / "dengue"

DATASET_FILES = {
    "colombia": DATA_DIRECTORY / "dengue_colombia.csv",
    "dengue_colombia": DATA_DIRECTORY / "dengue_colombia.csv",
    "panama": DATA_DIRECTORY / "dengue_panama.csv",
    "dengue_panama": DATA_DIRECTORY / "dengue_panama.csv",
}


def resolve_dataset_path(
    dataset: str | Path,
) -> tuple[Path, str]:
    """
    Resolve either a dengue dataset identifier or a direct CSV path.

    Supported identifiers:
        colombia
        dengue_colombia
        panama
        dengue_panama
    """
    if isinstance(dataset, Path):
        path = dataset
        dataset_id = path.stem
    else:
        key = dataset.strip().lower()

        if key in DATASET_FILES:
            path = DATASET_FILES[key]
            dataset_id = path.stem
        else:
            path = Path(dataset)
            dataset_id = path.stem

    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Dengue dataset not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, but received: {path}"
        )

    return path, dataset_id


def _validate_weekly_dates(
    dates: pd.Series,
) -> None:
    """
    Check that dates are valid, unique, ordered weekly observations.
    """
    if dates.isna().any():
        raise ValueError(
            "The time column contains invalid or missing dates."
        )

    if dates.duplicated().any():
        duplicate_dates = (
            dates[dates.duplicated()]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "The dataset contains duplicate dates: "
            f"{duplicate_dates[:5]}"
        )

    date_differences = dates.diff().dropna()

    if not date_differences.eq(pd.Timedelta(days=7)).all():
        bad_differences = (
            date_differences[
                ~date_differences.eq(pd.Timedelta(days=7))
            ]
            .head()
            .tolist()
        )

        raise ValueError(
            "The dengue dataset must contain consecutive weekly "
            "observations separated by 7 days. "
            f"Unexpected intervals: {bad_differences}"
        )


def _validate_values(
    values: pd.DataFrame,
) -> None:
    """Validate the regional dengue case-count columns."""

    if values.shape[1] < 1:
        raise ValueError(
            "The dataset must contain at least one regional series."
        )

    # Force all case-count columns to numeric values.
    try:
        numeric_values = values.apply(
            pd.to_numeric,
            errors="raise",
        )
    except Exception as exc:
        raise ValueError(
            "All dengue case-count columns must be numeric."
        ) from exc

    if numeric_values.isna().any().any():
        missing_count = int(
            numeric_values.isna().sum().sum()
        )
        raise ValueError(
            f"The dataset contains {missing_count} missing value(s)."
        )

    array = numeric_values.to_numpy(dtype=np.float64)

    if not np.isfinite(array).all():
        raise ValueError(
            "The dataset contains infinite or non-finite values."
        )

    if (array < 0).any():
        raise ValueError(
            "Dengue case counts must not be negative."
        )


def load_dengue_data(
    dataset: str | Path = "colombia",
    forecast_horizon: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load and split an EpiCastBench dengue dataset.

    The final `forecast_horizon` weekly observations are held out
    as the test set.

    Parameters
    ----------
    dataset:
        Dataset identifier such as "colombia" or "panama",
        or a direct CSV path.

    forecast_horizon:
        Number of final weekly observations to reserve for evaluation.

    Returns
    -------
    train_values:
        Array with shape
        (training_time_steps, number_of_regions).

    test_values:
        Hidden evaluation array with shape
        (forecast_horizon, number_of_regions).

    metadata:
        Dataset information including dates and region names.
    """
    if forecast_horizon < 1:
        raise ValueError(
            "forecast_horizon must be at least 1."
        )

    csv_path, dataset_id = resolve_dataset_path(dataset)

    dataframe = pd.read_csv(csv_path)

    if "time" not in dataframe.columns:
        raise ValueError(
            "The CSV file must contain a column named 'time'."
        )

    if dataframe.columns[0] != "time":
        raise ValueError(
            "The first CSV column must be named 'time'."
        )

    dates = pd.to_datetime(
        dataframe["time"],
        errors="coerce",
    )

    dataframe = dataframe.copy()
    dataframe["time"] = dates
    dataframe = (
        dataframe
        .sort_values("time")
        .reset_index(drop=True)
    )

    dates = dataframe["time"]

    values = (
        dataframe
        .drop(columns=["time"])
        .apply(pd.to_numeric, errors="raise")
    )

    _validate_weekly_dates(dates)
    _validate_values(values)

    number_of_rows = len(dataframe)

    if number_of_rows <= forecast_horizon:
        raise ValueError(
            "The dataset does not contain enough observations for "
            f"an {forecast_horizon}-week forecast."
        )

    minimum_training_length = max(
        2 * forecast_horizon,
        24,
    )

    training_length = (
        number_of_rows - forecast_horizon
    )

    if training_length < minimum_training_length:
        raise ValueError(
            "The training period is too short. "
            f"Expected at least {minimum_training_length} observations, "
            f"but found {training_length}."
        )

    all_values = values.to_numpy(dtype=np.float64)

    train_values = (
        all_values[:-forecast_horizon]
        .copy()
    )

    test_values = (
        all_values[-forecast_horizon:]
        .copy()
    )

    train_dates = dates.iloc[:-forecast_horizon]
    test_dates = dates.iloc[-forecast_horizon:]

    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "csv_path": str(csv_path),
        "frequency": "weekly",
        "locations": values.columns.astype(str).tolist(),
        "number_of_locations": int(values.shape[1]),
        "number_of_observations": int(number_of_rows),
        "training_observations": int(train_values.shape[0]),
        "forecast_horizon": int(forecast_horizon),
        "start_date": dates.iloc[0].date().isoformat(),
        "end_date": dates.iloc[-1].date().isoformat(),
        "train_start_date": (
            train_dates.iloc[0].date().isoformat()
        ),
        "train_end_date": (
            train_dates.iloc[-1].date().isoformat()
        ),
        "test_start_date": (
            test_dates.iloc[0].date().isoformat()
        ),
        "test_end_date": (
            test_dates.iloc[-1].date().isoformat()
        ),
        "train_dates": [
            date.date().isoformat()
            for date in train_dates
        ],
        "test_dates": [
            date.date().isoformat()
            for date in test_dates
        ],
    }

    return train_values, test_values, metadata


def describe_dengue_dataset(
    dataset: str | Path = "colombia",
    forecast_horizon: int = 8,
) -> dict[str, Any]:
    """
    Return a concise summary useful for command-line checks.
    """
    train_values, test_values, metadata = load_dengue_data(
        dataset=dataset,
        forecast_horizon=forecast_horizon,
    )

    return {
        "dataset_id": metadata["dataset_id"],
        "frequency": metadata["frequency"],
        "train_shape": tuple(train_values.shape),
        "test_shape": tuple(test_values.shape),
        "locations": metadata["locations"],
        "date_range": (
            metadata["start_date"],
            metadata["end_date"],
        ),
        "train_date_range": (
            metadata["train_start_date"],
            metadata["train_end_date"],
        ),
        "test_date_range": (
            metadata["test_start_date"],
            metadata["test_end_date"],
        ),
        "forecast_horizon": metadata["forecast_horizon"],
        "minimum_value": float(
            min(
                train_values.min(),
                test_values.min(),
            )
        ),
        "maximum_value": float(
            max(
                train_values.max(),
                test_values.max(),
            )
        ),
    }


if __name__ == "__main__":
    for dataset_name in (
        "colombia",
        "panama",
    ):
        summary = describe_dengue_dataset(
            dataset=dataset_name,
            forecast_horizon=8,
        )

        print()
        print("=" * 70)
        print(f"Dataset: {dataset_name}")
        print("=" * 70)

        for key, value in summary.items():
            print(f"{key}: {value}")
