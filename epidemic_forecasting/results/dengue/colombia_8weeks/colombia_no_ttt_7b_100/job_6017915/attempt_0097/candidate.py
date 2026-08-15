import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def preprocess_data(train_values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_values = scaler.fit_transform(train_values)
    return (scaled_train_values, scaler)

def postprocess_data(scaled_forecast, scaler):
    forecast = scaler.inverse_transform(scaled_forecast)
    return forecast.clip(0)

def sarimax_forecast(region_data, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    model = SARIMAX(region_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=horizon)
    return forecast

def exponential_smoothing_forecast(region_data, horizon, seasonal_periods=52):
    model = ExponentialSmoothing(region_data, seasonal_periods=seasonal_periods, trend='add', seasonal='add')
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    return forecast

def ensemble_forecast(region_data, horizon, n_estimators=10):
    forecasts = []
    for _ in range(n_estimators):
        if np.random.rand() < 0.5:
            forecast = sarimax_forecast(region_data, horizon)
        else:
            forecast = exponential_smoothing_forecast(region_data, horizon)
        forecasts.append(forecast)
    ensemble_forecast = np.mean(forecasts, axis=0)
    return ensemble_forecast

def dengue_forecast(train_values, horizon, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    _, n_regions = train_values.shape
    forecasts = []
    for i in range(n_regions):
        region_data = train_values[:, i]
        scaled_region_data, scaler = preprocess_data(region_data)
        forecast = ensemble_forecast(scaled_region_data, horizon)
        forecast = postprocess_data(forecast.reshape(-1, 1), scaler)
        forecasts.append(forecast.flatten())
    return np.array(forecasts).reshape(horizon, n_regions)