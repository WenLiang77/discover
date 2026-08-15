import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast Dengue incidence using ARIMA models for each region.
    
    :param train_values: np.ndarray of shape (T, N), where T is the number of historical time steps and N is the number of regions.
    :param horizon: int, the number of future time steps to predict.
    :return: np.ndarray of shape (horizon, N), containing the predicted Dengue incidences.
    """
    N = train_values.shape[1]
    forecasts = np.zeros((horizon, N))
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    for i in range(N):
        try:
            arima_model = ARIMA(train_scaled[:, i], order=(5, 1, 0))
            arima_model_fit = arima_model.fit(disp=False)
            forecast_scaled = arima_model_fit.forecast(steps=horizon)
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            forecast = np.maximum(forecast, 0)
            forecasts[:, i] = forecast
        except NotFittedError:
            ma_value = np.mean(train_values[-10:, i])
            forecasts[:, i] = ma_value
    return forecasts