import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
        except Exception as e:
            forecast[:, i] = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid')
            forecast = np.pad(forecast, ((0, horizon - len(forecast)), (0, 0)), mode='constant', constant_values=np.nan)
            forecast[:, i] = np.nan_to_num(forecast[:, i], nan=np.mean(train_values[:, i]), posinf=np.inf, neginf=-np.inf)
    forecast = np.maximum(forecast, 0)
    return forecast