import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            model_fit = model.fit(disp=False)
            forecast[:, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            forecast[:, i] = 0
    forecast = np.maximum(forecast, 0)
    return forecast