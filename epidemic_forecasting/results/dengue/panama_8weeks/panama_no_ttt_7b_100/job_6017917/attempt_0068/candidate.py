import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(train_values.shape[1])]
    for i in range(train_values.shape[1]):
        scalers[i].fit(train_values[:, i].reshape(-1, 1))
    forecasts = []
    for i in range(train_values.shape[1]):
        scaled_data = scalers[i].transform(train_values[:, i].reshape(-1, 1)).flatten()
        try:
            model = ARIMA(scaled_data, order=(5, 1, 0))
            model_fit = model.fit(disp=0)
            forecast_scaled = model_fit.forecast(steps=horizon)
            forecast = scalers[i].inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            forecast = np.maximum(forecast, 0)
            forecasts.append(forecast)
        except Exception as e:
            avg_value = np.mean(train_values[:, i])
            forecast = np.full(horizon, avg_value)
            forecast = np.maximum(forecast, 0)
            forecasts.append(forecast)
    return np.array(forecasts).T