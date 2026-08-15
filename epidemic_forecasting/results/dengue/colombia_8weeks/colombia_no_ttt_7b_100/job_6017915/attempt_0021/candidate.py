import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(0, 1, 1, 52))
            results = model.fit(disp=False)
            forecast[:, region] = results.forecast(steps=horizon)
        except Exception as e:
            window_size = min(2 * horizon, len(train_values[:, region]))
            forecast[:, region] = np.convolve(train_values[:, region], np.ones(window_size) / window_size, mode='valid')
            if len(forecast[:, region]) < horizon:
                forecast[-len(forecast[:, region]):, region] = forecast[-1, region]
    forecast = np.maximum(forecast, 0)
    return forecast