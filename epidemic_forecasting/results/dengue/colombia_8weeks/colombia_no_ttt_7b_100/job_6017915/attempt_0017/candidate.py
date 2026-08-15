import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.maximum(train_values, 0)
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ARIMA(train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit()
            forecast[:, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            forecast[:, i] = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid')
            forecast = np.pad(forecast, ((0, horizon - forecast.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
            forecast = np.nan_to_num(forecast, nan=0.0)
    forecast = np.maximum(forecast, 0)
    return forecast