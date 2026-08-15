import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence using ARIMA models for each region.
    
    Parameters:
    train_values (numpy.ndarray): Training data of shape (T, N), where T is the number of time steps and N is the number of regions.
    horizon (int): Number of future time steps to predict.
    kwargs (dict): Additional keyword arguments (not used).
    
    Returns:
    numpy.ndarray: Forecasted values of shape (horizon, N).
    """
    num_regions = train_values.shape[1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_values = scaler.fit_transform(train_values)
    forecasts = []
    for region in range(num_regions):
        try:
            arima_model = ARIMA(scaled_train_values[:, region], order=(5, 1, 0))
            arima_results = arima_model.fit(disp=False)
            forecast_scaled = arima_results.forecast(steps=horizon)
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))[:, 0]
            forecast = np.maximum(forecast, 0)
        except Exception as e:
            forecast = np.mean(train_values[-horizon:, region]) * np.ones(horizon)
            forecast = np.maximum(forecast, 0)
        forecasts.append(forecast)
    return np.array(forecasts).reshape(horizon, num_regions)