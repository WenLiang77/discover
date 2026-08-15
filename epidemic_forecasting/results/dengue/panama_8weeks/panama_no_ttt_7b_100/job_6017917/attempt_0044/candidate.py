import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast_values = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecast_values[:, region] = forecast
        except Exception as e:
            pass
    forecast_values = np.maximum(forecast_values, 0)
    return forecast_values