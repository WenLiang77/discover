import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecast = np.zeros((horizon, num_regions))
    for i in range(num_regions):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 52), enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            moving_average = np.convolve(train_values[:, i], np.ones(4) / 4, mode='valid')
            forecast[:len(moving_average), i] = moving_average
            forecast[len(moving_average):, i] = 0
    forecast = np.maximum(forecast, 0)
    return forecast