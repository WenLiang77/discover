import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict

def dengue_forecast(train_values: np.ndarray, horizon: int, **kwargs: Any) -> np.ndarray:
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values)
    for region in range(n_regions):
        try:
            arima_model = ARIMA(train_scaled[:, region], order=(1, 1, 0))
            arima_results = arima_model.fit()
            forecast_region = arima_results.get_forecast(steps=horizon).predicted_mean
            forecast_region = scaler.inverse_transform(forecast_region.reshape(-1, 1)).flatten()
            forecast_region = np.maximum(forecast_region, 0)
            forecast[:, region] = forecast_region
        except Exception as e:
            forecast[:, region] = np.full(horizon, train_values[-1, region])
    return forecast