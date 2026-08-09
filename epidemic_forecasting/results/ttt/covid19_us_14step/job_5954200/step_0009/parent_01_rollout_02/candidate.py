import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data += 1e-06
        try:
            arima_model = ARIMA(region_data, order=(1, 1, 1))
            arima_results = arima_model.fit()
            arima_aic = arima_results.aic
            sarimax_model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_results = sarimax_model.fit()
            sarimax_aic = sarimax_results.aic
            exponential_smoothing_model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
            exponential_smoothing_results = exponential_smoothing_model.fit()
            exponential_smoothing_aic = exponential_smoothing_results.aic
            if arima_aic < min(sarimax_aic, exponential_smoothing_aic):
                forecast = arima_results.forecast(steps=horizon)
            elif sarimax_aic < min(arima_aic, exponential_smoothing_aic):
                forecast = sarimax_results.forecast(steps=horizon)
            else:
                forecast = exponential_smoothing_results.forecast(steps=horizon)
        except Exception as _:
            forecast = np.full(horizon, np.mean(region_data))
        forecasted_values[:, region] = np.maximum(forecast, 0)
    return forecasted_values