import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        region_data = train_values[:, i]
        decomposition = seasonal_decompose(region_data, period=4, model='additive')
        trend_model = ExponentialSmoothing(decomposition.trend, trend='add', seasonal=None).fit()
        seasonal_model = ExponentialSmoothing(decomposition.seasonal, trend=None, seasonal='mul').fit()
        residual_model = ExponentialSmoothing(decomposition.resid, trend=None, seasonal=None).fit()
        trend_forecast = trend_model.forecast(steps=horizon)
        seasonal_forecast = seasonal_model.forecast(steps=horizon)
        residual_forecast = residual_model.forecast(steps=horizon)
        combined_forecast = trend_forecast + seasonal_forecast + residual_forecast
        forecast[:, i] = np.maximum(combined_forecast, 0)
    return forecast