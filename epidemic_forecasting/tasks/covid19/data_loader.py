from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPIDEMIC_DIRECTORY = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = EPIDEMIC_DIRECTORY / "data" / "covid19"

DATASET_FILES = {
    "uk": DATA_DIRECTORY / "covid_uk.csv",
    "covid_uk": DATA_DIRECTORY / "covid_uk.csv",
    "us": DATA_DIRECTORY / "covid_us.csv",
    "covid_us": DATA_DIRECTORY / "covid_us.csv",
}


def resolve_dataset_path(
    dataset: str | Path,
) -> tuple[Path, str]:
    """
    Resolve either a dataset identifier or a direct CSV path.

    Supported identifiers:
        uk
        covid_uk
        us
        covid_us
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
            f"COVID-19 dataset not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, but received: {path}"
        )

    return path, dataset_id


def _validate_daily_dates(
    dates: pd.Series,
) -> None:
    """Check that dates are unique and consecutive daily values."""
    if dates.isna().any():
        raise ValueError(
            "The time column contains invalid or missing dates."
        )

    if dates.duplicated().any():
        duplicate_dates = dates[dates.duplicated()].astype(str).tolist()
        raise ValueError(
            "The dataset contains duplicate dates: "
            f"{duplicate_dates[:5]}"
        )

    date_differences = dates.diff().dropna()

    if not date_differences.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "The COVID-19 dataset must contain consecutive daily dates."
        )


def _validate_values(
    values: pd.DataFrame,
) -> None:
    """Validate the regional case-count columns."""
    if values.shape[1] < 1:
        raise ValueError(
            "The dataset must contain at least one regional series."
        )

    if values.isna().any().any():
        missing_count = int(values.isna().sum().sum())
        raise ValueError(
            f"The dataset contains {missing_count} missing value(s)."
        )

    array = values.to_numpy(dtype=np.float64)

    if not np.isfinite(array).all():
        raise ValueError(
            "The dataset contains infinite or non-finite values."
        )

    if (array < 0).any():
        raise ValueError(
            "COVID-19 case counts must not be negative."
        )


def load_covid19_data(
    dataset: str | Path = "uk",
    forecast_horizon: int = 14,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load and split an EpiCastBench COVID-19 dataset.

    The final `forecast_horizon` rows are held out as the test set.

    Parameters
    ----------
    dataset:
        Dataset identifier such as "uk" or "us", or a direct CSV path.

    forecast_horizon:
        Number of final daily observations to reserve for evaluation.

    Returns
    -------
    train_values:
        Array with shape (training_time_steps, number_of_regions).

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
    dataframe = dataframe.sort_values("time").reset_index(drop=True)

    dates = dataframe["time"]
    values = dataframe.drop(columns=["time"])

    _validate_daily_dates(dates)
    _validate_values(values)

    number_of_rows = len(dataframe)

    if number_of_rows <= forecast_horizon:
        raise ValueError(
            "The dataset does not contain enough observations for "
            f"a {forecast_horizon}-step forecast."
        )

    minimum_training_length = max(
        2 * forecast_horizon,
        28,
    )

    training_length = number_of_rows - forecast_horizon

    if training_length < minimum_training_length:
        raise ValueError(
            "The training period is too short. "
            f"Expected at least {minimum_training_length} observations, "
            f"but found {training_length}."
        )

    all_values = values.to_numpy(dtype=np.float64)

    train_values = all_values[:-forecast_horizon].copy()
    test_values = all_values[-forecast_horizon:].copy()

    train_dates = dates.iloc[:-forecast_horizon]
    test_dates = dates.iloc[-forecast_horizon:]

    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "csv_path": str(csv_path),
        "frequency": "daily",
        "locations": values.columns.astype(str).tolist(),
        "number_of_locations": int(values.shape[1]),
        "number_of_observations": int(number_of_rows),
        "training_observations": int(train_values.shape[0]),
        "forecast_horizon": int(forecast_horizon),
        "start_date": dates.iloc[0].date().isoformat(),
        "end_date": dates.iloc[-1].date().isoformat(),
        "train_start_date": train_dates.iloc[0].date().isoformat(),
        "train_end_date": train_dates.iloc[-1].date().isoformat(),
        "test_start_date": test_dates.iloc[0].date().isoformat(),
        "test_end_date": test_dates.iloc[-1].date().isoformat(),
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


def describe_covid19_dataset(
    dataset: str | Path = "uk",
    forecast_horizon: int = 14,
) -> dict[str, Any]:
    """Return a concise summary useful for command-line checks."""
    train_values, test_values, metadata = load_covid19_data(
        dataset=dataset,
        forecast_horizon=forecast_horizon,
    )

    return {
        "dataset_id": metadata["dataset_id"],
        "train_shape": tuple(train_values.shape),
        "test_shape": tuple(test_values.shape),
        "locations": metadata["locations"],
        "date_range": (
            metadata["start_date"],
            metadata["end_date"],
        ),
        "forecast_horizon": metadata["forecast_horizon"],
        "minimum_value": float(
            min(train_values.min(), test_values.min())
        ),
        "maximum_value": float(
            max(train_values.max(), test_values.max())
        ),
    }


if __name__ == "__main__":
    for dataset_name in ("uk", "us"):
        summary = describe_covid19_dataset(
            dataset=dataset_name,
            forecast_horizon=14,
        )

        print(f"\nDataset: {dataset_name}")
        for key, value in summary.items():
            print(f"{key}: {value}")
