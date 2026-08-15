import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            alpha = 0.2
            smoothed_series = np.convolve(train_values[:, i], [alpha], mode='full')[:len(train_values[:, i])]
            forecast[:, i] = np.convolve(smoothed_series[::-1], [alpha], mode='valid')[::-1]
    forecast = np.maximum(forecast, 0)
    return forecast