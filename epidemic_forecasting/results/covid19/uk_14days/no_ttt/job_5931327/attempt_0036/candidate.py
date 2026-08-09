import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

def covid_forecast(train_values, horizon, **kwargs):
    """
    Forecast COVID-19 incidence data using a linear regression model.
    
    Parameters:
    train_values (np.ndarray): Training data with shape (T, N).
    horizon (int): Number of future time steps to predict.
    
    Returns:
    np.ndarray: Forecasted values with shape (horizon, N).
    """
    if train_values.size == 0:
        raise ValueError('Training data cannot be empty.')
    train_values = np.array(train_values)
    if np.any(train_values < 0):
        raise ValueError('Training data must contain only non-negative values.')
    T, N = train_values.shape
    model = MultiOutputRegressor(LinearRegression())
    try:
        model.fit(train_values[:-horizon], train_values[horizon:])
    except Exception as e:
        forecast = np.zeros((horizon, N))
        return forecast
    forecast = model.predict(train_values[-horizon:])
    forecast = np.maximum(forecast, 0)
    return forecast