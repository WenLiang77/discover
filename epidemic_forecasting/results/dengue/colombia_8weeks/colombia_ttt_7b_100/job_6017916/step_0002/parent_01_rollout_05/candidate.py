import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            decomposition = seasonal_decompose(train_values[:, i], model='additive', period=4)
            sarimax_model_trend = SARIMAX(decomposition.trend, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            sarimax_results_trend = sarimax_model_trend.fit(disp=False)
            forecast_trend = sarimax_results_trend.forecast(steps=horizon)
            sarimax_model_seasonal = SARIMAX(decomposition.seasonal, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            sarimax_results_seasonal = sarimax_model_seasonal.fit(disp=False)
            forecast_seasonal = sarimax_results_seasonal.forecast(steps=horizon)
            forecast[i] = forecast_trend + forecast_seasonal
            smoothing_level = 0.2
            es_model = ExponentialSmoothing(forecast[i], trend=None, seasonal=None, initialization_method='estimated')
            es_fit = es_model.fit(smoothing_level=smoothing_level)
            forecast[i] = es_fit.forecast(steps=horizon)
        except Exception as e:
            smoothing_level = 0.2
            es_model = ExponentialSmoothing(train_values[:, i], trend=None, seasonal=None, initialization_method='estimated')
            es_fit = es_model.fit(smoothing_level=smoothing_level)
            forecast[i] = es_fit.forecast(steps=horizon)
    forecast = np.maximum(forecast, 0)
    return forecast