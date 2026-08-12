from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Import MY evaluator from the discover repository
# ------------------------------------------------------------

DISCOVER = Path("discover-evaluator").resolve()
sys.path.insert(0, str(DISCOVER))

from epidemic_forecasting.tasks.covid19.evaluator import (
    calculate_metrics,
)


HORIZON = 14


def load_original_data(dataset):
    path = Path("data") / f"covid_{dataset}.csv"

    df = pd.read_csv(path)

    if "time" in df.columns:
        time_col = "time"
    else:
        time_col = df.columns[0]

    value_columns = [
        column
        for column in df.columns
        if column != time_col
    ]

    values = (
        df[value_columns]
        .to_numpy(dtype=np.float64)
    )

    train = values[:-HORIZON]
    actual = values[-HORIZON:]

    return train, actual, value_columns


def load_timesfm_predictions(dataset, value_columns):
    path = (
        Path("timesfm_results")
        / f"covid_{dataset}_timesfm_predictions.csv"
    )

    df = pd.read_csv(path, index_col=0)

    missing = [
        column
        for column in value_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"TimesFM predictions are missing columns: {missing}"
        )

    predictions = (
        df[value_columns]
        .to_numpy(dtype=np.float64)
    )

    return predictions


def load_epicastbench_metrics(dataset):
    path = (
        Path("timesfm_results")
        / f"covid_{dataset}_timesfm_metrics.csv"
    )

    df = pd.read_csv(path)

    row = {
        str(key).lower(): value
        for key, value in df.iloc[0].to_dict().items()
    }

    return row


def evaluate_dataset(dataset):
    train, actual, columns = load_original_data(dataset)

    predictions = load_timesfm_predictions(
        dataset,
        columns,
    )

    if predictions.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"prediction={predictions.shape}, "
            f"actual={actual.shape}"
        )

    my_metrics = calculate_metrics(
        train_values=train,
        actual=actual,
        predicted=predictions,
        mase_seasonality=1,
    )

    epicast_metrics = load_epicastbench_metrics(dataset)

    epicast_smape = float(
        epicast_metrics["smape"]
    )

    my_smape = float(
        my_metrics["smape"]
    )

    difference = abs(
        epicast_smape - my_smape
    )

    print()
    print("=" * 70)
    print(
        f"COVID-19 {dataset.upper()} — "
        f"TimesFM evaluator consistency check"
    )
    print("=" * 70)

    print("Shape:")
    print("  Train:      ", train.shape)
    print("  Actual:     ", actual.shape)
    print("  Prediction: ", predictions.shape)

    print()
    print("EpiCastBench / Darts:")
    print(
        f"  SMAPE = {epicast_smape:.12f}"
    )

    print()
    print("My discover evaluator:")
    print(
        f"  SMAPE = {my_smape:.12f}"
    )
    print(
        f"  MAE   = {my_metrics['mae']:.12f}"
    )
    print(
        f"  RMSE  = {my_metrics['rmse']:.12f}"
    )
    print(
        f"  MASE  = {my_metrics['mase']:.12f}"
    )

    print()
    print(
        f"Absolute SMAPE difference = "
        f"{difference:.12f}"
    )

    if difference < 1e-8:
        print(
            "RESULT: MATCH — the SMAPE implementations "
            "are numerically equivalent for this forecast."
        )
    elif difference < 1e-4:
        print(
            "RESULT: PRACTICAL MATCH — only negligible "
            "floating-point differences remain."
        )
    else:
        print(
            "RESULT: DIFFERENT — the evaluation pipelines "
            "need further investigation."
        )


evaluate_dataset("uk")
evaluate_dataset("us")
