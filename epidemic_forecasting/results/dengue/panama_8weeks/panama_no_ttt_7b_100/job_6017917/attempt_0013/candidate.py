import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        try:
            model = ARIMA(train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit()
            forecast[:, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            pass
    forecast = np.maximum(forecast, 0)
    return forecast