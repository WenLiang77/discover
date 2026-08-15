import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        series = train_values[:, i]
        try:
            model = ARIMA(series, order=(5, 1, 0))
            model_fit = model.fit(disp=False)
            forecast[:, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            try:
                model = SARIMAX(series, order=(2, 1, 0), seasonal_order=(1, 1, 0, 4))
                model_fit = model.fit(disp=False)
                forecast[:, i] = model_fit.forecast(steps=horizon)
            except Exception as e:
                mean = np.mean(series)
                forecast[:, i] = [mean] * horizon
    forecast = np.maximum(forecast, 0)
    return forecast