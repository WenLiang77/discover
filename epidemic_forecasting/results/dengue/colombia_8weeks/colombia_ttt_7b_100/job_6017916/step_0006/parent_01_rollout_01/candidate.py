import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            forecast[:, i] = np.full(horizon, train_values[-1, i])
    return forecast