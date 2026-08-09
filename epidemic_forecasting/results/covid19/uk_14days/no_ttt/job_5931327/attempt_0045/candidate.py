import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence data using an ARIMA model.
    
    Parameters:
    train_values (np.array): A 2D array of shape (T, N) containing historical incidence data.
    horizon (int): The number of future time steps to forecast.
    **kwargs: Additional keyword arguments (not used).
    
    Returns:
    np.array: A 2D array of shape (horizon, N) containing the forecasted incidence data.
    """
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = ARIMA(train_values[:, i], order=(5, 1, 0))
            model_fit = model.fit()
            forecast[:, i] = model_fit.forecast(steps=horizon)
        except Exception as e:
            ma_value = np.mean(train_values[:, i])
            forecast[:, i] = ma_value
    forecast[np.isinf(forecast)] = np.nanmean(forecast)
    forecast[np.isnan(forecast)] = 0
    forecast[forecast < 0] = 0
    return forecast