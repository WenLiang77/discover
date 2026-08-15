import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_values)
    forecasts = []
    for i in range(scaled_train.shape[1]):
        region_data = scaled_train[:, i]
        try:
            model = ARIMA(region_data, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).reshape(-1)
            forecast = np.clip(forecast, 0, None)
            forecasts.append(forecast)
        except Exception as e:
            avg_value = np.mean(region_data[-10:]) if len(region_data) >= 10 else 0
            forecast = [avg_value] * horizon
            forecasts.append(forecast)
    return np.array(forecasts).transpose()