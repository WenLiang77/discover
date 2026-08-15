import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import zscore

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            diff_series = train_values[:, i].diff().dropna()
            if len(diff_series) == 0:
                continue
            if (np.abs(zscore(diff_series)) < 3).all():
                model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            else:
                model = ARIMA(train_values[:, i], order=(1, 1, 0))
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='mul', seasonal_periods=4)
            results = model.fit(optimized=True)
            forecast[:, i] = results.forecast(steps=horizon)
    forecast = np.maximum(forecast, 0)
    return forecast