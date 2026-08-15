import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    T, N = train_values.shape
    forecast = np.zeros((horizon, N))
    for i in range(N):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            forecast[:, i] = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid')
            forecast[:, i] = np.pad(forecast[:, i], (0, horizon - len(forecast[:, i])), 'constant', constant_values=forecast[-1, i])
    forecast = np.maximum(forecast, 0)
    return forecast