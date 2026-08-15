import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        ts = train_values[:, i]
        try:
            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            forecast[:, i] = ts[-1] if len(ts) > 0 else 0
    forecast = np.maximum(forecast, 0)
    return forecast