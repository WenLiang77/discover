import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def covid_forecast(train_values, horizon, **kwargs):
    """
    A simple SARIMA-based forecasting model for multivariate COVID-19 incidence data.
    
    Parameters:
        train_values (np.ndarray): Historical training data with shape (T, N).
        horizon (int): Number of future time steps to predict.
        kwargs: Additional keyword arguments (not used in this implementation).
        
    Returns:
        np.ndarray: Predicted values with shape (horizon, N).
    """
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values[:, i], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            result = model.fit(disp=False)
            forecast[:horizon, i] = result.forecast(steps=horizon)
            forecast[:horizon, i] = np.maximum(forecast[:horizon, i], 0)
        except Exception as e:
            print(f'Failed to fit model for region {i}: {e}')
            forecast[:horizon, i] = 0
    return forecast