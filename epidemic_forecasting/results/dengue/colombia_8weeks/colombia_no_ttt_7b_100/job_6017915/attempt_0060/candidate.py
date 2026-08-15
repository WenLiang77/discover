import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_values = scaler.fit_transform(train_values)
    forecasts = []
    for region in range(scaled_train_values.shape[1]):
        region_data = scaled_train_values[:, region]
        try:
            model = ARIMA(region_data, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            forecast = scaler.inverse_transform(np.array([forecast]).T).flatten()
            forecast = np.maximum(forecast, 0)
        except Exception as e:
            avg_value = np.mean(region_data)
            forecast = np.full(horizon, avg_value)
        forecasts.append(forecast)
    return np.array(forecasts).reshape((horizon, -1))