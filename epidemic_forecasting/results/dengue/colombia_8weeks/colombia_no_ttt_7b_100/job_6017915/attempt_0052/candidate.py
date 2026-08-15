import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    T, N = train_values.shape
    forecast = np.zeros((horizon, N))
    for region in range(N):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecast[:, region] = results.forecast(steps=horizon)
        except Exception as e:
            alpha = 0.1
            smoothed_series = np.convolve(train_values[:, region], [alpha] * 4 + [1 - alpha * (4 + 1)], mode='valid')
            forecast[:len(smoothed_series), region] = smoothed_series
            forecast[len(smoothed_series):, region] = smoothed_series[-1]
    forecast[forecast < 0] = 0
    return forecast