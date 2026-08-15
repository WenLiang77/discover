import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import mae, rmse, smape

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        ts = train_values[:, i]
        result = adfuller(ts)
        if result[1] > 0.05:
            ts_diff = np.diff(ts)
        else:
            ts_diff = ts
        try:
            model = SARIMAX(ts_diff, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            model_fit = model.fit(disp=False)
            forecast_diff = model_fit.forecast(steps=horizon + len(ts_diff))
            forecast[i, :] = np.cumsum(forecast_diff[len(ts_diff):]) + ts[-1]
        except Exception as e:
            forecast[i, :] = np.mean(train_values, axis=0)[i]
    forecast = np.maximum(forecast, 0)
    return forecast