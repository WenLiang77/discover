import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(train_values.shape[1]):
        try:
            scaled_train = scaler.fit_transform(train_values[:, i].reshape(-1, 1))
            model = ExponentialSmoothing(scaled_train.flatten(), seasonal_periods=52, trend='add', seasonal='add')
            results = model.fit(use_boxcox=True)
            scaled_forecast = results.forecast(steps=horizon).reshape(1, -1)
            forecast[:, i] = scaler.inverse_transform(scaled_forecast)[0]
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            forecast[:, i] = np.full(horizon, np.mean(train_values[:, i][-5:], initial=0))
    return forecast