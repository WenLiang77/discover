import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence data using exponential smoothing.

    Parameters:
    train_values (np.ndarray): Training data of shape (T, N).
    horizon (int): Number of future time steps to predict.
    kwargs: Additional keyword arguments (not used).

    Returns:
    np.ndarray: Forecasted values of shape (horizon, N).
    """
    forecasts = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ExponentialSmoothing(train_values[:, i], trend='add', seasonal=None)
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(horizon)
            forecasts[:, i] = forecast
        except Exception as e:
            forecasts[:, i] = 0
    forecasts = np.maximum(forecasts, 0)
    return forecasts