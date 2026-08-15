import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            mask = train_values[:, i] > 0
            if not np.any(mask):
                continue
            train_data = train_values[mask, i]
            log_train_data = np.log(train_data)
            arima_model = ARIMA(log_train_data, order=(1, 1, 0))
            arima_results = arima_model.fit()
            forecast_arima = arima_results.forecast(steps=horizon)
            forecast[i] = np.exp(forecast_arima)
        except Exception as e:
            if train_values[:, i].std() == 0:
                forecast[i] = np.mean(train_values[:, i]) * np.ones(horizon)
            else:
                poly = PolynomialFeatures(degree=2)
                X = poly.fit_transform(np.arange(len(train_values))[:, None])
                y = train_values[:, i]
                ridge_model = Ridge(alpha=1.0)
                ridge_model.fit(X, y)
                X_forecast = poly.transform(np.arange(len(train_values), len(train_values) + horizon)[:, None])
                forecast_linear = ridge_model.predict(X_forecast)
                forecast[i] = np.maximum(forecast_linear, 0)
    return forecast