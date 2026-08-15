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
            scaled_forecast = results.get_forecast(steps=horizon).predicted_mean
            forecast[:, i] = scaler.inverse_transform(scaled_forecast.reshape(-1, 1)).flatten()
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            moving_avg = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid')
            forecast[:len(moving_avg), i] = moving_avg
            forecast[len(moving_avg):, i] = train_values[-1, i]
            forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast