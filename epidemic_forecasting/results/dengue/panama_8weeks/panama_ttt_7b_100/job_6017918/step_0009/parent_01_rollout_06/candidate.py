import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            train_values[:, i] = np.where(train_values[:, i] <= 0, np.nan, train_values[:, i])
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4), enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            if np.all(np.isnan(train_values[:, i])):
                regional_means = np.nanmean(train_values[:, train_values[:, i] > 0], axis=1)
                forecast[:, i] = np.interp(range(horizon), [0, len(regional_means) - 1], regional_means)
            else:
                X_train = np.arange(len(train_values)).reshape(-1, 1)
                y_train = train_values[:, i].reshape(-1, 1)
                lin_reg = LinearRegression().fit(X_train, y_train)
                X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
                forecast[:, i] = lin_reg.predict(X_forecast)
            forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast