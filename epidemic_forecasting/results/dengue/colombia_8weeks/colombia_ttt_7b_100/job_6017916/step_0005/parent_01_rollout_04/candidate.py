import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(train_values.shape[1]):
        try:
            scaled_train = scaler.fit_transform(train_values[:, i].reshape(-1, 1))
            model = ARIMA(scaled_train, order=(1, 1, 0))
            results = model.fit(disp=False)
            scaled_forecast = results.forecast(steps=horizon).reshape(-1, 1)
            forecast[:, i] = scaler.inverse_transform(scaled_forecast)
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecast[:, i] = np.full(horizon, train_values[-1, i])
    return forecast