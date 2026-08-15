import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal='add', seasonal_periods=52)
            fitted_model = model.fit(use_boxcox=False, remove_bias=True)
            forecast[:, i] = fitted_model.forecast(steps=horizon).clip(0)
        except Exception as e:
            forecast[:, i] = np.convolve(train_values[:, i], np.ones(horizon) / horizon, mode='valid').repeat(horizon // len(train_values[:, i]) + 1)[:horizon]
            forecast[:, i] = forecast[:, i].clip(0)
    return forecast