import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from math import isnan, inf

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]), dtype=float)
    pt = PowerTransformer(method='yeo-johnson')
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        if np.any(region_data <= 0):
            region_data += 1
        log_data = np.log(region_data)
        transformed_data = pt.fit_transform(log_data.reshape(-1, 1)).flatten()
        try:
            decomposition = seasonal_decompose(transformed_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            arima_model = SARIMAX(trend, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            arima_fitted_model = arima_model.fit(disp=False)
            trend_forecast = arima_fitted_model.get_forecast(steps=horizon).predicted_mean
            es_model = ExponentialSmoothing(seasonal, trend='add', seasonal_periods=7)
            es_fitted_model = es_model.fit()
            seasonal_forecast = es_fitted_model.forecast(steps=horizon)
            combined_forecast = (np.exp(trend_forecast + seasonal_forecast) - 1) * pt.inverse_transform(np.ones((horizon, 1)), fitted=True).flatten()
            forecasted_values[:, region] = np.maximum(combined_forecast, 0)
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                es_fitted_model = es_model.fit()
                forecast = es_fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values