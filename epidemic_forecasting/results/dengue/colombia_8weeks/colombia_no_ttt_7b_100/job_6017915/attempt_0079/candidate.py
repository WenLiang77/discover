import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52)
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecasts[:, i] = forecast
        except Exception as e:
            moving_avg = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid')
            padded_avg = np.pad(moving_avg, (0, horizon - len(moving_avg)), 'edge')
            forecasts[:, i] = padded_avg
    return forecasts