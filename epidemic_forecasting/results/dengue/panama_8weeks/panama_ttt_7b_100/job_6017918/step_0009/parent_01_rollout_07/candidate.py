import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if np.any(train_values[:, i] <= 0):
                train_values[:, i] = np.where(train_values[:, i] <= 0, np.nan, train_values[:, i])
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4), enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            scaler = np.std(train_values[:, i]) if np.std(train_values[:, i]) != 0 else 1e-06
            X_train = np.arange(len(train_values)).reshape(-1, 1)
            y_train = train_values[:, i].reshape(-1, 1)
            slope = np.dot(X_train - np.mean(X_train), y_train - np.mean(y_train)) / ((X_train - np.mean(X_train)) ** 2).sum()
            intercept = np.mean(y_train) - slope * np.mean(X_train)
            X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
            forecast[:, i] = slope * X_forecast + intercept
            forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast