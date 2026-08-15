import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_values)
    forecasts = []
    for i in range(scaled_train.shape[1]):
        region_series = scaled_train[:, i]
        try:
            model = SARIMAX(region_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon).predicted_mean
            forecast_original_scale = scaler.inverse_transform(forecast.reshape(-1, 1))[:, 0]
            forecast_original_scale = np.maximum(forecast_original_scale, 0)
            forecasts.append(forecast_original_scale)
        except Exception as e:
            constant_forecast = np.full(horizon, np.mean(region_series, axis=0))
            forecast_original_scale = np.maximum(constant_forecast, 0)
            forecasts.append(forecast_original_scale)
    forecasts_array = np.array(forecasts).T
    return forecasts_array