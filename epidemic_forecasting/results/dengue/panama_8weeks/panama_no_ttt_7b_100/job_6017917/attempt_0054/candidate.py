import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecasts dengue incidence using a SARIMAX model for each region.

    Args:
        train_values: A NumPy array of shape (T, N) containing historical
                      dengue incidence data.
        horizon: An integer indicating the number of future time steps to predict.

    Returns:
        A NumPy array of shape (horizon, N) containing the predicted dengue
        incidence values.
    """
    num_regions = train_values.shape[1]
    predictions = np.zeros((horizon, num_regions))
    for i in range(num_regions):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(0, 1, 1, 52))
            model_fit = model.fit(disp=False)
            forecast_result = model_fit.get_forecast(steps=horizon)
            forecast = forecast_result.predicted_mean
            forecast = np.maximum(forecast, 0)
            predictions[:, i] = forecast
        except Exception as e:
            window_size = min(10, len(train_values))
            rolling_avg = np.convolve(train_values[:, i], np.ones(window_size) / window_size, mode='valid')
            padding = np.full(horizon - len(rolling_avg), rolling_avg[-1])
            fallback_predictions = np.concatenate((rolling_avg, padding))
            predictions[:, i] = fallback_predictions
    return predictions