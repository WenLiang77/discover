import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(train_values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_values)
    return (scaled_train, scaler)

def fit_model(scaled_train, horizon):
    forecasts = []
    for i in range(scaled_train.shape[1]):
        region_series = scaled_train[:, i]
        if len(region_series) > horizon * 2:
            try:
                model = ExponentialSmoothing(region_series, seasonal_periods=52, trend='add', seasonal='add').fit()
                forecast = model.forecast(steps=horizon).clip(0, None)
                forecasts.append(forecast)
            except Exception as e:
                avg_value = np.mean(region_series)
                forecast = np.full(horizon, avg_value)
                forecasts.append(forecast)
        else:
            avg_value = np.mean(region_series)
            forecast = np.full(horizon, avg_value)
            forecasts.append(forecast)
    return np.array(forecasts)

def dengue_forecast(train_values, horizon, **kwargs):
    scaled_train, scaler = preprocess_data(train_values)
    forecasts = fit_model(scaled_train, horizon)
    unscaled_forecasts = scaler.inverse_transform(forecasts)
    return unscaled_forecasts