import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if is_stationary(train_values[:, i]):
                model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            else:
                model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='mul', seasonal_periods=4)
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            forecast[:, i] = train_values[-1, i]
    forecast = np.maximum(forecast, 0)
    return forecast

def is_stationary(series):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series)
    return result[1] < 0.05