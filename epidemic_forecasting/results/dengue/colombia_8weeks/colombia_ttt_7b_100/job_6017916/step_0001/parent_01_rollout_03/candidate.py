import numpy as np
from sklearn.linear_model import ARIMA
from sklearn.multioutput import MultiOutputRegressor

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ARIMA(train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit()
            forecast[:, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            forecast[:, i] = 0
    forecast = np.maximum(forecast, 0)
    return forecast