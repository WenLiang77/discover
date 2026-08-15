import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecast = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast[:, region] = results.forecast(steps=horizon)
        except Exception as e:
            window_size = min(52, len(train_values))
            rolling_mean = np.convolve(train_values[:, region], np.ones(window_size) / window_size, mode='valid')
            forecast[:len(rolling_mean), region] = rolling_mean
            if horizon > len(rolling_mean):
                forecast[len(rolling_mean):, region] = train_values[-1, region]
    return np.clip(forecast, 0, None)