import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from math import isnan, inf

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]), dtype=float)
    pt = PowerTransformer(method='yeo-johnson')
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        log_data = np.log(region_data + 1)
        decomposition = seasonal_decompose(log_data, model='additive', period=7)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        try:
            arima_trend_model = ARIMA(trend, order=(1, 1, 1))
            arima_trend_fit = arima_trend_model.fit(disp=False)
            trend_forecast = arima_trend_fit.forecast(steps=horizon)
            es_seasonal_model = ExponentialSmoothing(seasonal, trend=None, seasonal_periods=7)
            es_seasonal_fit = es_seasonal_model.fit()
            seasonal_forecast = es_seasonal_fit.forecast(steps=horizon)
            combined_forecast = np.exp(trend_forecast + seasonal_forecast) - 1
            forecasted_values[:, region] = np.maximum(combined_forecast, 0)
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(log_data, trend='add', seasonal=None)
                es_fit = es_model.fit()
                forecast_log = es_fit.forecast(steps=horizon)
                forecast = np.exp(forecast_log) - 1
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values