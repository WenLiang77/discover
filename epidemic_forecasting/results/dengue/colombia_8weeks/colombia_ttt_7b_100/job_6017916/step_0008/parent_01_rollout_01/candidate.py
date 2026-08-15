import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(train_values.shape[1]):
        try:
            scaled_train = scaler.fit_transform(train_values[:, i].reshape(-1, 1))
            model = ExponentialSmoothing(scaled_train.flatten(), trend='add', seasonal='add', seasonal_periods=52)
            fit_model = model.fit(optimized=True, remove_nan=True)
            scaled_forecast = fit_model.forecast(steps=horizon).reshape(-1, 1)
            forecast[:, i] = scaler.inverse_transform(scaled_forecast).flatten()
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            forecast[:, i] = np.full(horizon, train_values[-1, i])
    return forecast