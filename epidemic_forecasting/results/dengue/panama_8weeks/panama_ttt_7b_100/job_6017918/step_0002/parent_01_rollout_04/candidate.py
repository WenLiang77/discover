import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            try:
                model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='add', seasonal_periods=4)
                results = model.fit()
                forecast[:, i] = results.forecast(steps=horizon)
            except Exception as e:
                forecast[:, i] = train_values[-1, i]
    forecast = np.maximum(forecast, 0)
    return forecast