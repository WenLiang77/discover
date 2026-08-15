import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(train_values.shape[1]):
        try:
            scaled_train = scaler.fit_transform(train_values[:, i].reshape(-1, 1))
            try:
                arima_model = ARIMA(scaled_train.flatten(), order=(5, 1, 0))
                arima_results = arima_model.fit(disp=False)
                scaled_arima_forecast = arima_results.forecast(steps=horizon)
                forecast[:, i] = scaler.inverse_transform(scaled_arima_forecast.reshape(-1, 1)).flatten()
                forecast[:, i] = np.maximum(forecast[:, i], 0)
            except Exception:
                pass
            if np.any(np.isnan(forecast[:, i])):
                exponential_smoothing_model = ExponentialSmoothing(scaled_train.flatten(), trend='add', seasonal='mul', seasonal_periods=52)
                exponential_smoothing_results = exponential_smoothing_model.fit(disp=False)
                scaled_es_forecast = exponential_smoothing_results.forecast(steps=horizon)
                forecast[:, i] = scaler.inverse_transform(scaled_es_forecast.reshape(-1, 1)).flatten()
                forecast[:, i] = np.maximum(forecast[:, i], 0)
            if np.any(np.isnan(forecast[:, i])):
                forecast[:, i] = np.full(horizon, train_values[-1, i])
        except Exception as e:
            forecast[:, i] = np.full(horizon, train_values[-1, i])
    return forecast