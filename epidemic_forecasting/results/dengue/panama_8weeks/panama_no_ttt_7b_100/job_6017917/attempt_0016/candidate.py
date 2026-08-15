import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values)
    N = train_values.shape[1]
    forecasts = []
    for i in range(N):
        ts = train_scaled[:, i]
        try:
            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        except Exception as e:
            moving_avg = np.convolve(ts, np.ones(horizon) / horizon, mode='valid')
            forecasts.append(np.concatenate((moving_avg, [moving_avg[-1]] * (horizon - len(moving_avg)))))
            continue
        forecast = model.forecast(steps=horizon)
        forecast_original_scale = scaler.inverse_transform(forecast.reshape(-1, 1))[:, 0]
        forecast_original_scale = np.maximum(forecast_original_scale, 0)
        forecasts.append(forecast_original_scale)
    forecasts_array = np.array(forecasts).reshape(horizon, N)
    return forecasts_array