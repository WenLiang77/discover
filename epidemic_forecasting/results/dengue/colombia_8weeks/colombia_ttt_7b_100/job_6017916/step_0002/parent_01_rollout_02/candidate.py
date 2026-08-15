import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ARIMA(train_values[:, i], order=(1, 1, 0))
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            forecast[:, i] = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid')
            forecast = np.pad(forecast, ((0, horizon - len(forecast)), (0, 0)), mode='edge')
    forecast = np.maximum(forecast, 0)
    return forecast