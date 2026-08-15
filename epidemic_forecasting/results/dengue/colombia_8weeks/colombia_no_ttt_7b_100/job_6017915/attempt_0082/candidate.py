import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        series = train_values[:, i]
        result = adfuller(series)
        if result[1] > 0.05:
            series_diff = np.diff(series)
            series_diff = np.append(series_diff, [np.nan])
        else:
            series_diff = series
        try:
            model = SARIMAX(series_diff[:-1], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
            results = model.fit(disp=False)
            forecast_steps = horizon + len(series_diff)
            forecast_series_diff = results.forecast(steps=forecast_steps)[-horizon:]
            forecast[i] = np.cumsum(forecast_series_diff)
            forecast[i][0] = series[-1]
            forecast[i] = np.maximum(forecast[i], 0)
        except Exception as e:
            forecast[i] = np.convolve(series, np.ones(horizon) / horizon, mode='valid')
            forecast[i] = np.pad(forecast[i], (0, horizon - len(forecast[i])), 'constant', constant_values=(0, 0))
            forecast[i] = np.maximum(forecast[i], 0)
    return forecast