import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, i] = forecast.clip(0)
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecasted_values[:, i] = 0
    return forecasted_values