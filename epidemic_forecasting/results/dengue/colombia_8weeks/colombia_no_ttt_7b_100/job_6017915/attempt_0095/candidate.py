import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        region_data = train_values[:, i]
        try:
            model = SARIMAX(region_data, order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecasts[:, i] = forecast.clip(0)
        except Exception as e:
            moving_avg = np.convolve(region_data, np.ones(horizon) / horizon, mode='valid')
            padded_avg = np.pad(moving_avg, (0, horizon - len(moving_avg)), mode='edge')
            forecasts[:, i] = padded_avg[:horizon].clip(0)
    return forecasts