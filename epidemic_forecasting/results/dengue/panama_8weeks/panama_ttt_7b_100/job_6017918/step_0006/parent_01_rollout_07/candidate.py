import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import RidgeCV

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        if np.any(train_values[:, i] <= 0):
            train_values[:, i] = np.where(train_values[:, i] <= 0, np.nan, train_values[:, i])
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4), enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
        except Exception as e:
            if np.all(np.isnan(train_values[:, i])):
                regional_means = np.nanmean(train_values[:, train_values[:, i] > 0], axis=1)
                forecast[:, i] = np.interp(range(horizon), [0, len(regional_means) - 1], regional_means)
            else:
                X_train = np.arange(len(train_values)).reshape(-1, 1)
                y_train = train_values[:, i].reshape(-1, 1)
                ridge_cv = RidgeCV(cv=5)
                ridge_cv.fit(X_train, y_train.ravel())
                X_forecast = np.arange(len(train_values), len(train_values) + horizon).reshape(-1, 1)
                forecast[:, i] = ridge_cv.predict(X_forecast)
    forecast = np.maximum(forecast, 0)
    return forecast