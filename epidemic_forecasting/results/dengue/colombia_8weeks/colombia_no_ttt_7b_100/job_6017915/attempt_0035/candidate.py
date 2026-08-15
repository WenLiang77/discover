import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, n_regions), dtype=np.float32)
    for i in range(n_regions):
        region_data = train_values[:, i]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(region_data.reshape(-1, 1))
        try:
            model = SARIMAX(scaled_data.flatten(), order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecast_scaled = scaler.inverse_transform(forecast.reshape(-1, 1))
            forecasted_values[:, i] = np.clip(forecast_scaled.flatten(), 0, None)
        except Exception as e:
            forecasted_values[:, i] = np.mean(region_data) * np.ones(horizon)
    return forecasted_values