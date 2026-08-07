import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
            forecast = np.maximum(forecast, 0)
            forecasts[:, i] = forecast
        except Exception as e:
            forecasts[:, i] = np.zeros(horizon)
    return forecasts