import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def covid_forecast(train_values, horizon, **kwargs):
    forecasts = np.zeros((horizon, train_values.shape[1]))
    random_state = kwargs.get('random_state', None)
    for i in range(train_values.shape[1]):
        try:
            if (np.diff(train_values[:, i]) != 0).any():
                model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='mul', seasonal_periods=7)
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecasts[:, i] = forecast
        except Exception as e:
            sma = np.mean(train_values[:, i])
            forecasts[:, i] = sma
    return np.clip(forecasts, 0, None)