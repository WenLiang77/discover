import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    random_state = kwargs.get('random_state', None)
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            forecast[:, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            if train_values.shape[0] > 1:
                slope = (train_values[-1, i] - train_values[0, i]) / (train_values.shape[0] - 1)
                forecast[:, i] = np.arange(1, horizon + 1) * slope + train_values[-1, i]
            else:
                forecast[:, i] = train_values[-1, i]
    forecast = np.maximum(forecast, 0)
    return forecast