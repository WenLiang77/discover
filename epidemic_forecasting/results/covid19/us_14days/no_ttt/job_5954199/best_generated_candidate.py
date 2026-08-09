import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, region] = forecast
        except Exception as e:
            moving_avg = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
            forecasted_values[:len(moving_avg), region] = moving_avg
    return np.maximum(forecasted_values, 0)