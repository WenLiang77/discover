import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        region_data = train_values[:, i]
        if len(region_data) > 0:
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(region_data.reshape(-1, 1))
                model = SARIMAX(scaled_data.flatten(), order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
                results = model.fit(disp=False)
                forecast_scaled = results.forecast(steps=horizon).reshape(1, -1)
                forecast_unscaled = scaler.inverse_transform(forecast_scaled)
                forecasts[:, i] = np.maximum(forecast_unscaled, 0)
            except Exception as e:
                ma_value = np.mean(region_data[-10:])
                forecasts[:, i] = np.full(horizon, max(ma_value, 0))
        else:
            forecasts[:, i] = np.zeros(horizon)
    return forecasts