import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def dengue_forecast(train_values, horizon, random_state=None):
    """
    Forecast Dengue incidence using an ARIMA model for each region.

    Parameters:
    train_values (np.ndarray): Shape (T, N), where T is the number of historical time steps
                              and N is the number of geographical regions.
    horizon (int): Number of future time steps to predict.
    random_state (int, optional): Seed for reproducibility.

    Returns:
    np.ndarray: Forecasted values with shape (horizon, N).
    """
    scaler = StandardScaler()
    scaler.fit(train_values)
    scaled_train = scaler.transform(train_values)
    forecasts = []
    for region_data in scaled_train.T:
        try:
            model = ARIMA(region_data, order=(1, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            inverse_forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
            inverse_forecast = np.maximum(inverse_forecast, 0)
            forecasts.append(inverse_forecast)
        except Exception as e:
            moving_avg = np.convolve(region_data, np.ones(horizon) / horizon, mode='valid')
            padded_avg = np.pad(moving_avg, (0, horizon - len(moving_avg)), 'edge')
            inverse_forecast = scaler.inverse_transform(padded_avg.reshape(-1, 1))
            inverse_forecast = np.maximum(inverse_forecast, 0)
            forecasts.append(inverse_forecast)
    forecast_array = np.array(forecasts).T
    return forecast_array