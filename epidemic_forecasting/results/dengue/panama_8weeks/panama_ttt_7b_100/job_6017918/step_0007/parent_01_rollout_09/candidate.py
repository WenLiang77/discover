import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            decomposition = seasonal_decompose(train_values[:, i], model='additive', period=52)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            arima_model = ARIMA(residual, order=(1, 1, 0))
            arima_results = arima_model.fit(disp=False)
            forecast_residuals = arima_results.forecast(steps=horizon)
            forecast_trend = np.tile(trend[-1], horizon) if trend is not None else np.zeros(horizon)
            forecast_seasonal = np.tile(seasonal[-1], horizon) if seasonal is not None else np.zeros(horizon)
            forecast[i, :] = forecast_trend + forecast_seasonal + forecast_residuals
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(train_values[:, i], seasonal_periods=52, trend=None, seasonal='add')
                es_results = es_model.fit(smoothing_level=0.2, optimized=False)
                forecast[i, :] = es_results.forecast(steps=horizon)
            except Exception as e:
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
                forecast[i, :] = scaler.inverse_transform(forecast_scaled)
        forecast[i, :] = np.maximum(forecast[i, :], 0)
    return forecast