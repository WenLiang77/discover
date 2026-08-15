import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence for multiple regions using ARIMA, SARIMAX, and Exponential Smoothing.

    Parameters:
        train_values (np.ndarray): Training data of shape (T, N).
        horizon (int): Number of future time steps to predict.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        np.ndarray: Forecasted values of shape (horizon, N).
    """
    train_values = np.array(train_values)
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        region_data = train_values[:, i]
        try:
            arima_model = ARIMA(region_data, order=(5, 1, 0)).fit(disp=False)
            arima_forecast = arima_model.forecast(steps=horizon)
            forecasts[:len(arima_forecast), i] += arima_forecast
        except Exception as e:
            pass
        try:
            sarimax_model = SARIMAX(region_data, order=(5, 1, 0), seasonal_order=(1, 1, 0, 7)).fit(disp=False)
            sarimax_forecast = sarimax_model.forecast(steps=horizon)
            forecasts[:len(sarimax_forecast), i] += sarimax_forecast
        except Exception as e:
            pass
        try:
            es_model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=7).fit(disp=False)
            es_forecast = es_model.forecast(steps=horizon)
            forecasts[:len(es_forecast), i] += es_forecast
        except Exception as e:
            pass
    forecasts /= 3
    forecasts = np.maximum(forecasts, 0)
    return forecasts