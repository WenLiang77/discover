import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if np.any(train_values[:, i] <= 0):
                train_values[:, i] = np.where(train_values[:, i] <= 0, np.nan, train_values[:, i])
            if np.abs(np.diff(train_values[:, i], n=2)).mean() < 1e-05:
                scaler = StandardScaler()
                X_train = np.arange(len(train_values)).reshape(-1, 1)
                y_train = train_values[:, i].reshape(-1, 1)
                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                y_train_scaled = scaler.transform(y_train)
                lin_reg = LinearRegression()
                lin_reg.fit(X_train_scaled, y_train_scaled)
                X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
                X_forecast_scaled = scaler.transform(X_forecast)
                forecast_scaled = lin_reg.predict(X_forecast_scaled)
                forecast[:, i] = scaler.inverse_transform(forecast_scaled)
            else:
                model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4), enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
                forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            if np.all(np.isnan(train_values[:, i])):
                regional_means = np.nanmean(train_values[:, train_values[:, i] > 0], axis=1)
                forecast[:, i] = np.interp(range(horizon), [0, len(regional_means) - 1], regional_means)
            else:
                scaler = StandardScaler()
                X_train = np.arange(len(train_values)).reshape(-1, 1)
                y_train = train_values[:, i].reshape(-1, 1)
                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                y_train_scaled = scaler.transform(y_train)
                lin_reg = LinearRegression()
                lin_reg.fit(X_train_scaled, y_train_scaled)
                X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
                X_forecast_scaled = scaler.transform(X_forecast)
                forecast_scaled = lin_reg.predict(X_forecast_scaled)
                forecast[:, i] = scaler.inverse_transform(forecast_scaled)
            forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast