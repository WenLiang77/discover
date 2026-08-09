import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from math import isnan, inf

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]), dtype=float)
    scaler = MinMaxScaler(feature_range=(1, np.max(train_values) * 10))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        scaled_region_data = scaler.fit_transform(region_data.reshape(-1, 1)).flatten()
        try:
            es_model = ExponentialSmoothing(scaled_region_data, trend='add', seasonal=None)
            es_fitted_model = es_model.fit(smoothing_level=0.2, optimized=False)
            es_forecast = es_fitted_model.forecast(steps=horizon)
            arima_model = ARIMA(scaled_region_data, order=(1, 1, 1))
            arima_fitted_model = arima_model.fit()
            arima_forecast = arima_fitted_model.forecast(steps=horizon)
            combined_forecast = (es_forecast + arima_forecast) / 2
            inverse_forecast = scaler.inverse_transform(combined_forecast.reshape(-1, 1)).flatten()
            forecasted_values[:, region] = np.clip(inverse_forecast, 0, None)
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(scaled_region_data, trend='add', seasonal=None)
                es_fitted_model = es_model.fit(smoothing_level=0.2, optimized=False)
                forecasted_values[:, region] = np.clip(scaler.inverse_transform(es_fitted_model.forecast(steps=horizon).reshape(-1, 1)), 0, None).flatten()
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values