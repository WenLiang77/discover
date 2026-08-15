import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecast = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        time_series = train_values[:, region]
        try:
            model = ARIMA(time_series, order=(5, 1, 0))
            model_fit = model.fit(disp=False)
            forecast[:, region] = model_fit.forecast(steps=horizon)
        except Exception as e:
            forecast[:, region] = np.mean(time_series)
    forecast = np.maximum(forecast, 0)
    return forecast