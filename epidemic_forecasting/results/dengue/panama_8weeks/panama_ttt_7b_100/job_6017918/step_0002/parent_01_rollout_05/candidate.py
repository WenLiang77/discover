import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            acf_vals = plot_acf(train_values[:, i], lags=20, ax=None)
            pacf_vals = plot_pacf(train_values[:, i], lags=20, ax=None)
            arima_model = ARIMA(train_values[:, i], order=(1, 1, 0))
            arima_results = arima_model.fit()
            sarimax_model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            sarimax_results = sarimax_model.fit(disp=False)
            arima_forecast = arima_results.forecast(steps=horizon)
            sarimax_forecast = sarimax_results.forecast(steps=horizon)
            combined_forecast = (arima_forecast + sarimax_forecast) / 2
            combined_forecast = np.maximum(combined_forecast, 0)
            forecast[:, i] = combined_forecast
        except Exception as e:
            es_model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='multiplicative')
            es_results = es_model.fit(disp=False)
            forecast[:, i] = es_results.forecast(steps=horizon)
    return forecast