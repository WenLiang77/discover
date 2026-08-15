import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        ts = train_values[:, i]
        try:
            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            moving_avg = np.convolve(ts, np.ones(horizon) / horizon, mode='valid')
            forecast[:len(moving_avg), i] = moving_avg
        forecast[:, i] = np.maximum(forecast[:, i], 0)
    return forecast