import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def dengue_forecast(train_values, horizon, random_state=None):
    train_values = np.array(train_values)
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        ts = train_values[:, i]
        result = adfuller(ts)
        is_stationary = result[1] < 0.05
        if not is_stationary:
            ts_diff = np.diff(ts)
            ts_diff = np.insert(ts_diff, 0, 0)
        else:
            ts_diff = ts
        try:
            model = SARIMAX(ts_diff, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            forecast_diff = results.forecast(steps=horizon)
            if not is_stationary:
                forecast[i, :] = np.cumsum(forecast_diff) + ts[0]
            else:
                forecast[i, :] = forecast_diff
        except Exception as e:
            forecast[i, :] = np.convolve(ts, np.ones(horizon) / horizon, mode='valid')
            if len(forecast[i, :]) < horizon:
                forecast[i, :] = np.mean(ts) * np.ones(horizon)
    forecast = np.maximum(forecast, 0)
    return forecast