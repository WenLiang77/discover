import numpy as np


def covid_forecast(train_values, horizon, **kwargs):
    train_values = np.asarray(train_values, dtype=float)

    if train_values.ndim != 2:
        raise ValueError("train_values must be a two-dimensional array.")

    if train_values.shape[0] < 1:
        raise ValueError("At least one historical observation is required.")

    last_value = train_values[-1]
    forecast = np.repeat(last_value[None, :], int(horizon), axis=0)

    return np.clip(forecast, 0.0, None)
