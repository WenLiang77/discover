import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    scaler = StandardScaler()
    for i in range(train_values.shape[1]):
        try:
            scaled_train = scaler.fit_transform(train_values[:, i].reshape(-1, 1))
            model = SARIMAX(scaled_train, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            scaled_forecast = results.forecast(steps=horizon)
            forecast[:, i] = scaler.inverse_transform(scaled_forecast.reshape(-1, 1)).flatten()
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            window_size = min(10, len(train_values))
            moving_avg = np.convolve(train_values[:, i], np.ones(window_size) / window_size, mode='valid')
            extended_avg = np.concatenate(([moving_avg[0]] * (window_size - 1), moving_avg, [moving_avg[-1]] * (horizon - len(moving_avg))))
            forecast[:len(extended_avg), i] = extended_avg
    return forecast