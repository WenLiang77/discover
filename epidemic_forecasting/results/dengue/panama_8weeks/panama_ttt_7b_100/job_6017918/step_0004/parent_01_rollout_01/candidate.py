import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from typing import Optional

def check_stationarity(timeseries: np.ndarray) -> bool:
    result = adfuller(timeseries)
    return result[1] < 0.05

def make_stationary(timeseries: np.ndarray) -> np.ndarray:
    _, transformed, _ = boxcox(timeseries + 1)
    return transformed

def inverse_transform(timeseries: np.ndarray, transformed_timeseries: np.ndarray) -> np.ndarray:
    inverse_transformed, _ = boxcox(transformed_timeseries, lmbda=None)
    return inverse_transformed - 1

def fit_and_predict(model_class, timeseries: np.ndarray, horizon: int, **fit_args):
    model = model_class(timeseries, **fit_args)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=horizon).predicted_mean
    return forecast

def dengue_forecast(train_values: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
    num_regions = train_values.shape[1]
    forecast = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if not check_stationarity(region_data):
            region_data = make_stationary(region_data)
        try:
            forecast[:, region] = fit_and_predict(SARIMAX, region_data, horizon, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
        except Exception as e:
            forecast[:, region] = train_values[-1, region]
        forecast[:, region] = np.maximum(forecast[:, region], 0)
        if not check_stationarity(train_values[:, region]):
            forecast[:, region] = inverse_transform(train_values[:, region], forecast[:, region])
    return forecast