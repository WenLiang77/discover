import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    forecast = np.zeros((horizon, train_values.shape[1]))
    scaler = StandardScaler()
    for i in range(train_values.shape[1]):
        try:
            scaled_train = scaler.fit_transform(train_values[:, i].reshape(-1, 1))
            model = ExponentialSmoothing(scaled_train.flatten(), trend='add', seasonal='add', seasonal_periods=4).fit(use_boxcox=True, disp=False)
            scaled_forecast = model.forecast(steps=horizon)
            forecast[:, i] = scaler.inverse_transform(scaled_forecast.reshape(-1, 1)).flatten()
            forecast[:, i] = np.maximum(forecast[:, i], 0)
        except Exception as e:
            forecast[:, i] = np.full(horizon, train_values[-1, i])
    return forecast