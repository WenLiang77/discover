import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(5, 1, 0), seasonal_order=(1, 1, 1, 4))
            model_fit = model.fit(disp=False)
            forecast[:horizon, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            window_size = min(20, len(train_values))
            rolling_mean = np.convolve(train_values[:, i], np.ones(window_size) / window_size, mode='valid')
            forecast[:len(rolling_mean), i] = rolling_mean
            last_value = train_values[-1, i]
            forecast[len(rolling_mean):, i] = last_value
    forecast = np.maximum(forecast, 0)
    return forecast