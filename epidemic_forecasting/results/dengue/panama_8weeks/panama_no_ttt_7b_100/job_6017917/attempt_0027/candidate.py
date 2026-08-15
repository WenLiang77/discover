import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_values)
    forecasts = []
    for region in range(scaled_train.shape[1]):
        try:
            model = SARIMAX(scaled_train[:, region], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecast_unscaled = scaler.inverse_transform(np.array([forecast]).T)
            forecast_unscaled = np.maximum(forecast_unscaled, 0)
            forecasts.append(forecast_unscaled.flatten())
        except Exception as e:
            avg_value = np.mean(train_values[-1, region])
            forecast_unscaled = np.full(horizon, avg_value).reshape(-1, 1)
            forecasts.append(np.maximum(forecast_unscaled, 0).flatten())
    return np.vstack(forecasts).reshape(horizon, train_values.shape[1])