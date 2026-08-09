import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    pt = PowerTransformer(method='yeo-johnson')
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data += 1e-06
        region_data_pt = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
        region_data_scaled = scaler.fit_transform(region_data_pt.reshape(-1, 1)).flatten()
        try:
            model = SARIMAX(region_data_scaled, enforce_stationarity=False, enforce_invertibility=False)
            fitted_model = model.fit(disp=False)
            forecast_scaled = fitted_model.get_forecast(steps=horizon).predicted_mean
        except Exception as _:
            try:
                model = ExponentialSmoothing(region_data_scaled, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast_scaled = fitted_model.forecast(steps=horizon)
            except Exception as _:
                forecast_scaled = np.full(horizon, np.median(region_data_scaled))
        forecast_original_scale = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        forecast_original_scale = pt.inverse_transform(forecast_original_scale.reshape(-1, 1)).flatten()
        forecasted_values[:, region] = np.maximum(forecast_original_scale, 0)
    return forecasted_values