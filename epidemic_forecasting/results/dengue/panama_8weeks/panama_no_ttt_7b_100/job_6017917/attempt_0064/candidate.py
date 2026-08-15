import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            result = model.fit(disp=False)
            forecast[:horizon, region] = result.forecast(steps=horizon)
        except Exception as e:
            forecast[:horizon, region] = np.convolve(train_values[:, region], np.ones(horizon) / horizon, mode='valid')
    forecast = np.maximum(forecast, 0)
    return forecast