import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if np.isnan(train_values[-1, i]):
                window_size = 5
                moving_avg = np.convolve(train_values[:len(train_values) - window_size, i], np.ones(window_size) / window_size, mode='valid')
                forecast[:len(moving_avg), i] = moving_avg
                forecast[len(moving_avg):, i] = train_values[-window_size, i]
            else:
                model = ARIMA(train_values[:, i], order=(1, 1, 1))
                results = model.fit()
                forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            forecast[:, i] = train_values[-1, i]
    forecast = np.maximum(forecast, 0)
    return forecast