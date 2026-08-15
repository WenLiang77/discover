import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1] < 0.05

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if not check_stationarity(train_values[:, i]):
                diffed_series = np.diff(train_values[:, i])
                model = SARIMAX(diffed_series, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
                results = model.fit(disp=False)
                forecast_diff = results.forecast(steps=horizon)
                forecast[i] = np.cumsum(forecast_diff) + train_values[-1, i]
            else:
                model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
                results = model.fit(disp=False)
                forecast[:, i] = results.forecast(steps=horizon)
            forecast[i] = np.maximum(forecast[i], 0)
        except Exception as e:
            forecast[:, i] = np.mean(train_values, axis=0)
    return forecast