import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(train_values):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_values)
    return (scaled_data, scaler)

def fit_models(scaled_data):
    models = []
    for i in range(scaled_data.shape[1]):
        model = ExponentialSmoothing(scaled_data[:, i], trend='add', seasonal='add', seasonal_periods=52).fit(use_boxcox=True)
        models.append(model)
    return models

def forecast(models, scaler, horizon):
    forecast_values = np.zeros((horizon, len(models)))
    for i, model in enumerate(models):
        forecast_values[:, i] = model.forecast(horizon)
    forecast_values = scaler.inverse_transform(forecast_values)
    return np.maximum(forecast_values, 0)

def dengue_forecast(train_values, horizon, **kwargs):
    scaled_data, scaler = preprocess_data(train_values)
    models = fit_models(scaled_data)
    forecast_values = forecast(models, scaler, horizon)
    return forecast_values