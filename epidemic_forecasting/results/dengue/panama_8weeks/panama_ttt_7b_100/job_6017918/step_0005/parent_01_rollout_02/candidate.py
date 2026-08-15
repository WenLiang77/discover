import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if np.any(train_values[:, i] <= 0):
                train_values[:, i] = np.where(train_values[:, i] <= 0, np.nan, train_values[:, i])
            arima_model = ARIMA(train_values[:, i], order=(1, 1, 0))
            arima_results = arima_model.fit()
            forecast_arima = arima_results.forecast(steps=horizon)
            if np.any(np.isnan(forecast_arima)):
                sarimax_model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
                sarimax_results = sarimax_model.fit()
                forecast_sarimax = sarimax_results.forecast(steps=horizon)
                if np.any(np.isnan(forecast_sarimax)):
                    scaler = StandardScaler()
                    X_train = np.arange(len(train_values)).reshape(-1, 1)
                    y_train = train_values[:, i].reshape(-1, 1)
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    y_train_scaled = scaler.transform(y_train)
                    ridge_model = Ridge(alpha=1.0)
                    ridge_model.fit(X_train_scaled, y_train_scaled)
                    X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
                    X_forecast_scaled = scaler.transform(X_forecast)
                    forecast_ridge = ridge_model.predict(X_forecast_scaled)
                    forecast[:, i] = scaler.inverse_transform(forecast_ridge)
                else:
                    forecast[:, i] = forecast_sarimax
            else:
                forecast[:, i] = forecast_arima
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            print(f'Error during forecasting for region {i}: {e}')
            regional_means = np.nanmean(train_values[:, train_values[:, i] > 0], axis=1)
            forecast[:, i] = np.interp(range(horizon), [0, len(regional_means) - 1], regional_means)
    return forecast