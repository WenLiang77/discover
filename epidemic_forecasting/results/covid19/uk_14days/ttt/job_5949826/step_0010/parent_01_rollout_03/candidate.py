import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from math import isnan, inf

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]), dtype=float)
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        if len(region_data) < 2:
            forecasted_values[:, region] = 0
            continue
        log_data = np.log(region_data + 1)
        try:
            decomposition = seasonal_decompose(log_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            arima_model_trend = ARIMA(trend, order=(1, 1, 1))
            arima_fitted_model_trend = arima_model_trend.fit(disp=False)
            trend_forecast = arima_fitted_model_trend.get_forecast(steps=horizon).predicted_mean
            es_model_seasonal = ExponentialSmoothing(seasonal, trend='add', seasonal_periods=7)
            es_fitted_model_seasonal = es_model_seasonal.fit()
            seasonal_forecast = es_fitted_model_seasonal.forecast(steps=horizon)
            combined_forecast = np.exp(trend_forecast + seasonal_forecast) - 1
            forecasted_values[:, region] = np.maximum(combined_forecast, 0)
        except Exception as e:
            try:
                arima_model_direct = ARIMA(log_data, order=(1, 1, 1))
                arima_fitted_model_direct = arima_model_direct.fit(disp=False)
                direct_forecast = arima_fitted_model_direct.get_forecast(steps=horizon).predicted_mean
                forecasted_values[:, region] = np.maximum(np.exp(direct_forecast) - 1, 0)
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values