import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    scaler = MinMaxScaler()
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        scaled_data = scaler.fit_transform(train_values[:, region].reshape(-1, 1)).flatten()
        try:
            model = ExponentialSmoothing(scaled_data, seasonal_periods=52, trend='add', seasonal='add')
            fitted_model = model.fit(disp=False)
            forecast_scaled = fitted_model.forecast(steps=horizon)
        except Exception as e:
            forecast_scaled = np.full(horizon, np.mean(scaled_data))
        forecasted_values[:, region] = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    return forecasted_values.clip(min=0)