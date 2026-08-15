import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        region_data = train_values[:, i]
        if np.all(region_data == region_data[0]):
            forecast[:, i] = region_data[0]
            continue
        try:
            model = ExponentialSmoothing(region_data, trend='add', seasonal=None, seasonal_periods=None)
            fit_model = model.fit()
            forecast[:, i] = fit_model.forecast(steps=horizon).clip(0)
        except Exception as e:
            moving_avg = np.convolve(region_data, np.ones(horizon) / horizon, mode='valid')
            if len(moving_avg) < horizon:
                moving_avg = np.full(horizon, np.mean(region_data)).clip(0)
            forecast[:len(moving_avg), i] = moving_avg
    return forecast