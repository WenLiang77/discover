import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence using ARIMA models for each region.
    
    Parameters:
    train_values (np.ndarray): Training data with shape (T, N).
    horizon (int): Number of future time steps to predict.
    
    Returns:
    np.ndarray: Forecasted values with shape (horizon, N).
    """
    T, N = train_values.shape
    forecasted_values = np.zeros((horizon, N))
    for i in range(N):
        ts = train_values[:, i]
        try:
            model = ARIMA(ts, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, i] = forecast
        except Exception as e:
            forecasted_values[:, i] = np.mean(ts[-20:], axis=0)
    return forecasted_values