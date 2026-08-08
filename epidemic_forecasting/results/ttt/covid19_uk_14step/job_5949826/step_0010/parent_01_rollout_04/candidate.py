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
        log_data = np.log(region_data + 1)
        try:
            decomposition = seasonal_decompose(log_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            arima_model = ARIMA(trend, order=(1, 1, 1))
            arima_fitted_model = arima_model.fit(disp=False)
            trend_forecast = arima_fitted_model.forecast(steps=horizon)
            es_model = ExponentialSmoothing(seasonal, trend='add', seasonal_periods=7)
            es_fitted_model = es_model.fit()
            seasonal_forecast = es_fitted_model.forecast(steps=horizon)
            combined_forecast = np.exp(trend_forecast + seasonal_forecast) - 1
            forecasted_values[:, region] = np.clip(combined_forecast, 0, None)
        except Exception as e:
            try:
                arima_model = ARIMA(region_data, order=(1, 1, 1))
                arima_fitted_model = arima_model.fit(disp=False)
                forecast = arima_fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.clip(forecast, 0, None)
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values