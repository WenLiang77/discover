import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    forecasts = []
    for i in range(train_values.shape[1]):
        region_data = train_scaled[:, i]
        try:
            model = SARIMAX(region_data, order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon).predicted_mean
            forecast_original_scale = scaler.inverse_transform(forecast.reshape(-1, 1))[:, 0]
            forecast_original_scale = np.maximum(forecast_original_scale, 0)
            forecasts.append(forecast_original_scale)
        except Exception as e:
            avg_value = np.mean(region_data)
            forecast_original_scale = np.full(horizon, avg_value)
            forecasts.append(np.maximum(forecast_original_scale, 0))
    return np.array(forecasts).reshape(horizon, -1)