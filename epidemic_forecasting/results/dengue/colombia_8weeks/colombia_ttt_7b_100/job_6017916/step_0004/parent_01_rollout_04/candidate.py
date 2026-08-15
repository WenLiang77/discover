import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            if np.allclose(train_values[:, i], train_values[:, i][0]):
                forecast[:, i] = train_values[-1, i]
            else:
                model = SARIMAX(train_values[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
                results = model.fit(disp=False)
                forecast[:, i] = results.forecast(steps=horizon)
        except Exception as e:
            model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='add', seasonal_periods=4)
            fit_model = model.fit(use_box_cox=True, disp=False)
            forecast[:, i] = fit_model.forecast(steps=horizon)
    forecast = np.maximum(forecast, 0)
    return forecast