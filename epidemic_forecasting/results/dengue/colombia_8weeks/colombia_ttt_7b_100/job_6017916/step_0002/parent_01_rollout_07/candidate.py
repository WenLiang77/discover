import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            arima_model = ARIMA(train_values[:, i], order=(1, 1, 0))
            arima_results = arima_model.fit()
            sarimax_model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            sarimax_results = sarimax_model.fit()
            linreg_model = LinearRegression()
            X = np.arange(len(train_values)).reshape(-1, 1)
            linreg_model.fit(X, train_values[:, i])
            rf_model = RandomForestRegressor(n_estimators=10, random_state=kwargs.get('random_state'))
            rf_model.fit(X, train_values[:, i])
            forecasts = [arima_results.forecast(steps=horizon), sarimax_results.forecast(steps=horizon), linreg_model.predict(np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)), rf_model.predict(np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1))]
            combined_forecast = np.mean(forecasts, axis=0)
            forecast[:, i] = np.maximum(combined_forecast, 0)
        except Exception as e:
            forecast[:, i] = train_values[-1, i]
    return forecast