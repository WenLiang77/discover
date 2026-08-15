import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52)
    num_regions = train_values.shape[1]
    forecast_horizon = horizon
    forecasts = np.zeros((forecast_horizon, num_regions))
    for i in range(num_regions):
        region_series = train_values[:, i]
        try:
            model = SARIMAX(region_series, order=order, seasonal_order=seasonal_order)
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=forecast_horizon).predicted_mean
            forecast = np.maximum(forecast, 0)
            forecasts[:, i] = forecast
        except Exception as e:
            moving_avg = np.convolve(region_series, np.ones(forecast_horizon) / forecast_horizon, mode='valid')
            forecast = np.pad(moving_avg, (0, forecast_horizon - len(moving_avg)), 'constant', constant_values=(moving_avg[-1], 0))
            forecast = np.maximum(forecast, 0)
            forecasts[:, i] = forecast
    return forecasts