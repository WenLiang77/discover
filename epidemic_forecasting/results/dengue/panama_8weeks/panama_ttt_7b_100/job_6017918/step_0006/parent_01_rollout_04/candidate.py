import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            train_values[:, i] = np.where(train_values[:, i] <= 0, np.nan, train_values[:, i])
            model = ARIMA(train_values[:, i], order=(1, 1, 0))
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            if np.all(np.isnan(train_values[:, i])):
                regional_means = np.nanmean(train_values[:, train_values[:, i] > 0], axis=1)
                forecast[:, i] = np.interp(range(horizon), [0, len(regional_means) - 1], regional_means)
            else:
                scaler = MinMaxScaler(feature_range=(0, np.max(train_values[:, i])))
                X_train = np.arange(len(train_values)).reshape(-1, 1)
                y_train = train_values[:, i].reshape(-1, 1)
                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                y_train_scaled = scaler.transform(y_train)
                lin_reg = np.polyfit(X_train_scaled.flatten(), y_train_scaled.flatten(), 1)
                X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
                X_forecast_scaled = scaler.transform(X_forecast)
                forecast_scaled = lin_reg[0] * X_forecast_scaled + lin_reg[1]
                forecast[:, i] = scaler.inverse_transform(forecast_scaled)
        forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast