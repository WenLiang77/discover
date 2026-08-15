import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence for multiple regions using SARIMAX models.
    
    Parameters:
        train_values (np.ndarray): Training data with shape (T, N).
        horizon (int): Number of future time steps to predict.
        kwargs (dict): Additional keyword arguments (not used).

    Returns:
        np.ndarray: Forecasted values with shape (horizon, N).
    """
    T, N = train_values.shape
    scaler = MinMaxScaler()
    scaled_train_values = scaler.fit_transform(train_values)
    forecasts = []
    for i in range(N):
        try:
            model = SARIMAX(scaled_train_values[:, i], order=(1, 1, 0), seasonal_order=(0, 1, 0, 52))
            result = model.fit(disp=False)
            forecast = result.get_forecast(steps=horizon).predicted_mean
            forecast = scaler.inverse_transform(forecast.reshape(-1, 1))[:, 0]
            forecast = np.maximum(forecast, 0)
        except NotFittedError:
            forecast = np.mean(scaled_train_values[:, i][-10:], axis=0)
            forecast = np.repeat([forecast], horizon, axis=0)
            forecast = scaler.inverse_transform(forecast.reshape(-1, 1))[:, 0]
            forecast = np.maximum(forecast, 0)
        forecasts.append(forecast)
    return np.array(forecasts).reshape(horizon, N)