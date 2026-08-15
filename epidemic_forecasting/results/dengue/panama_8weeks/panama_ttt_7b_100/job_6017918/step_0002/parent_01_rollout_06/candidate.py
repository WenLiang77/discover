import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    if result[1] <= 0.05:
        return True
    else:
        return False

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if check_stationarity(train_values[:, i]):
                order = (1, 0, 0)
                seasonal_order = (0, 1, 0, 4)
            else:
                order = (1, 1, 0)
                seasonal_order = (1, 1, 0, 4)
            model = SARIMAX(train_values[:, i], order=order, seasonal_order=seasonal_order)
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            alpha = 0.2
            forecast[:, i] = np.convolve(train_values[:, i], [alpha] * (1 + int(1 / alpha)), mode='valid')[-horizon:]
    forecast = np.maximum(forecast, 0)
    return forecast