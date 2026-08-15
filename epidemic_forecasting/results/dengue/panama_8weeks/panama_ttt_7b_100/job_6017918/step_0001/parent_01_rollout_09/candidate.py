import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
        except Exception as e:
            alpha = 0.1
            smoothed_series = np.convolve(train_values[:, i], [alpha] * 52, mode='valid')
            smoothed_series = np.pad(smoothed_series, (0, horizon - len(smoothed_series)), 'edge')
            forecast[:, i] = smoothed_series
    return np.clip(forecast, 0, None)