import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    T, N = train_values.shape
    freq = kwargs.get('frequency', 'D')
    forecasts = np.zeros((horizon, N))
    for i in range(N):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 7), enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecasts[:, i] = forecast.clip(0)
        except Exception as e:
            forecasts[:, i] = 0
            print(f'Failed to fit model for region {i}: {e}')
    return forecasts