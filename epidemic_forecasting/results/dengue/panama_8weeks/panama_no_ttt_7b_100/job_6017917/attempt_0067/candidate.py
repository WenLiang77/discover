import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence using a SARIMAX model for each region.

    Parameters:
        train_values (np.ndarray): Training data with shape (T, N).
        horizon (int): Number of future time steps to predict.
        kwargs: Additional keyword arguments (not used).

    Returns:
        np.ndarray: Forecasted values with shape (horizon, N).
    """
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecasts[:, region] = np.maximum(forecast, 0)
        except Exception as e:
            moving_avg = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
            padded_moving_avg = np.pad(moving_avg, ((0, horizon - len(moving_avg)), (0, 0)), mode='edge')
            forecasts[:, region] = np.maximum(padded_moving_avg, 0)
    return forecasts