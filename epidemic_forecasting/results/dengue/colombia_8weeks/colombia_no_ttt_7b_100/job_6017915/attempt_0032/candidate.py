import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence using SARIMA model for each region.

    Args:
        train_values (np.ndarray): Training data with shape (T, N).
        horizon (int): Number of future time steps to predict.
        kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: Forecasted values with shape (horizon, N).
    """
    n_regions = train_values.shape[1]
    forecasts = []
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(2, 1, 0), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
        except Exception as e:
            rolling_mean = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
            forecast = np.full((horizon,), rolling_mean[-1])
            forecasts.append(forecast)
            continue
        forecast = results.forecast(steps=horizon)
        forecast = np.maximum(forecast, 0)
        forecasts.append(forecast)
    return np.array(forecasts).reshape((horizon, n_regions))