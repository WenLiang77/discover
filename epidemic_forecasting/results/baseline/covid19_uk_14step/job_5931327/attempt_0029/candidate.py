import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            model_fit = model.fit(disp=False)
        except Exception as e:
            model_fit = None
        if model_fit is not None:
            forecast[:, i] = model_fit.forecast(steps=horizon)
        forecast[:, i] = np.clip(forecast[:, i], 0, None)
    return forecast