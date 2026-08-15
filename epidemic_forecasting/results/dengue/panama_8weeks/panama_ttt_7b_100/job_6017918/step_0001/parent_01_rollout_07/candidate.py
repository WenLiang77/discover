import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon).predicted_mean
            forecasts[:, region] = forecast.clip(0)
        except Exception as e:
            rolling_mean = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
            forecasts[:len(rolling_mean), region] = np.pad(rolling_mean, (0, horizon - len(rolling_mean)), 'edge').clip(0)
    return forecasts