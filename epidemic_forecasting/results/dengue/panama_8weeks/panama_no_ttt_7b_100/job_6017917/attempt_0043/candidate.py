import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ARIMA(train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit()
            forecast[:, i] = model_fit.forecast(steps=horizon)
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            print(f'Failed to fit ARIMA model for region {i}: {e}')
            forecast[:, i] = 0
    return forecast