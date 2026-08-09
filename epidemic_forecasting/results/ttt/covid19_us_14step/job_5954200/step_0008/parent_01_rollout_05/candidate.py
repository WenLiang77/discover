import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    scaler = StandardScaler(with_mean=False)
    imputer = SimpleImputer(strategy='mean')
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data += 1e-06
        region_data_filled = imputer.fit_transform(region_data.reshape(-1, 1)).flatten()
        region_data_scaled = scaler.fit_transform(region_data_filled.reshape(-1, 1)).flatten()
        try:
            model = SARIMAX(region_data_scaled, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast_scaled = fitted_model.get_forecast(steps=horizon).predicted_mean
        except Exception as _:
            try:
                model = ExponentialSmoothing(region_data_scaled, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast_scaled = fitted_model.forecast(steps=horizon)
            except Exception as _:
                forecast_scaled = np.full(horizon, np.mean(region_data_scaled))
        forecast_original_scale = np.maximum(scaler.inverse_transform(forecast_scaled.reshape(-1, 1)), 0).flatten()
        forecasted_values[:, region] = forecast_original_scale
    return forecasted_values