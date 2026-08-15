import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict

def preprocess_data(train_values: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(train_values)

def postprocess_data(forecast: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.inverse_transform(forecast)

def dengue_forecast(train_values: np.ndarray, horizon: int, **kwargs: Any) -> np.ndarray:
    if train_values.shape[1] == 0:
        raise ValueError('Input data is empty.')
    if train_values.min() < 0:
        raise ValueError('Input data contains negative values.')
    if not np.isfinite(train_values).all():
        raise ValueError('Input data contains non-finite values.')
    num_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, num_regions))
    scalers = [MinMaxScaler() for _ in range(num_regions)]
    for region in range(num_regions):
        region_data = preprocess_data(train_values[:, region].reshape(-1, 1))
        try:
            model = ARIMA(region_data, order=(5, 1, 0))
            model_fit = model.fit(disp=0)
            forecast_steps = model_fit.forecast(steps=horizon)
            forecasts[:, region] = postprocess_data(np.array([forecast_steps]).T, scalers[region])
        except Exception as e:
            forecast_mean = np.mean(train_values[:, region], axis=0)
            forecast_std = np.std(train_values[:, region], axis=0)
            forecast_steps = forecast_mean + np.random.randn(horizon) * forecast_std
            forecast_steps = np.clip(forecast_steps, 0, None)
            forecasts[:, region] = forecast_steps
    return forecasts