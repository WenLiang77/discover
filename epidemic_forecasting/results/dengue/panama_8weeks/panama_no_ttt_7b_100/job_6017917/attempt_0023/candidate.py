import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_values)
    forecasts = []
    for i in range(scaled_train.shape[1]):
        series = scaled_train[:, i]
        try:
            model = SARIMAX(series, order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
        except Exception as e:
            forecast = np.mean(series[-10:], axis=0) * np.ones(horizon)
        unscaled_forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))[:, 0]
        unscaled_forecast = np.maximum(unscaled_forecast, 0)
        forecasts.append(unscaled_forecast)
    return np.array(forecasts).reshape(horizon, -1)