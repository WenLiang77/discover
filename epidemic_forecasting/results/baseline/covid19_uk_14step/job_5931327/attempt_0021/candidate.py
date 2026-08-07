import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence data using SARIMAX model.

    :param train_values: 2D NumPy array of historical training data with shape (T, N)
    :param horizon: Number of future time steps to predict
    :return: 2D NumPy array of forecasted values with shape (horizon, N)
    """
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast[i, :] = fitted_model.forecast(steps=horizon)
            forecast[i, :] = np.maximum(forecast[i, :], 0)
        except Exception as e:
            forecast[i, :] = 0
    return forecast