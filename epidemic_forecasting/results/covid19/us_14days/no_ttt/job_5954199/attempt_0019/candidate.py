import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    A simple SARIMA-based forecasting function for COVID-19 incidence data.
    
    Args:
    train_values (np.ndarray): Training data with shape (T, N).
    horizon (int): Number of future time steps to predict.
    
    Returns:
    np.ndarray: Forecasted values with shape (horizon, N).
    """
    train_values = np.array(train_values)
    forecast_values = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            result = model.fit(disp=False)
            forecast = result.get_forecast(steps=horizon).predicted_mean
            forecast_values[:, i] = np.maximum(forecast, 0)
        except Exception as e:
            forecast_values[:, i] = 0
            print(f'Failed to fit model for region {i}: {e}')
    return forecast_values