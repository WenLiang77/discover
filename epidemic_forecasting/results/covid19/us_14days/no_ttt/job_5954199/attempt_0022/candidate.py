import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def covid_forecast(train_values, horizon, **kwargs):
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        try:
            model = ExponentialSmoothing(train_values[:, region], trend='add', seasonal=None)
            fitted_model = model.fit()
            forecast_region = fitted_model.forecast(steps=horizon)
            forecast_region = np.maximum(forecast_region, 0)
            forecasts[:, region] = forecast_region
        except Exception as e:
            forecasts[:, region] = 0
    return forecasts