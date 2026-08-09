import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)

def postprocess_data(data, scaler):
    return scaler.inverse_transform(data)

def detect_seasonality(data):
    decomposition = seasonal_decompose(data, model='additive')
    return decomposition.seasonal

def detect_trend(data):
    decomposition = seasonal_decompose(data, model='additive')
    return decomposition.trend

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        scaled_data = preprocess_data(region_data.reshape(-1, 1)).flatten()
        seasonal = detect_seasonality(scaled_data)
        trend = detect_trend(scaled_data)
        combined_data = scaled_data + seasonal + trend
        try:
            model = SARIMAX(combined_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
            forecast = postprocess_data(forecast.reshape(-1, 1), MinMaxScaler(feature_range=(0, 1))).flatten()
            forecasted_values[:, region] = np.maximum(forecast, 0)
        except Exception as _:
            pass
        if np.allclose(forecasted_values[:, region], 0):
            try:
                model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as _:
                pass
    return forecasted_values