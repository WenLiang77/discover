import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from math import isnan, inf

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]), dtype=float)
    pt = PowerTransformer(method='yeo-johnson')
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        transformed_data = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
        decomposition = seasonal_decompose(transformed_data, model='additive', period=7)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        try:
            arima_model = ARIMA(trend, order=(1, 1, 1))
            arima_fitted_model = arima_model.fit(disp=False)
            trend_forecast = arima_fitted_model.forecast(steps=horizon)
            es_model = ExponentialSmoothing(seasonal, trend='add', seasonal_periods=7)
            es_fitted_model = es_model.fit()
            seasonal_forecast = es_fitted_model.forecast(steps=horizon)
            combined_forecast = np.exp(trend_forecast + seasonal_forecast) - 1
            original_scale_forecast = pt.inverse_transform(combined_forecast.reshape(-1, 1)).flatten()
            forecasted_values[:, region] = np.maximum(original_scale_forecast, 0)
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(transformed_data, trend='add', seasonal=None)
                es_fitted_model = es_model.fit()
                forecast = es_fitted_model.forecast(steps=horizon)
                original_scale_forecast = pt.inverse_transform(forecast.reshape(-1, 1)).flatten()
                forecasted_values[:, region] = np.maximum(original_scale_forecast, 0)
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values