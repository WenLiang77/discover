import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, random_state=42):
    train_values = np.array(train_values)
    forecast = np.zeros((horizon, train_values.shape[1]))
    np.random.seed(random_state)
    for i in range(train_values.shape[1]):
        try:
            model = ARIMA(train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit()
            forecast[:model_fit.order[0], i] = np.nan
            forecast[model_fit.order[0]:, i] = model_fit.forecast(steps=horizon - model_fit.order[0])
        except Exception as e:
            forecast[:, i] = np.nan_to_num(forecast[max(0, horizon - len(train_values)), i])
    forecast = np.maximum(forecast, 0)
    return forecast