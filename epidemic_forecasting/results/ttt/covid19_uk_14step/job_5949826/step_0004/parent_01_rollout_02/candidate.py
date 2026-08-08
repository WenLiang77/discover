import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def covid_forecast(train_values, horizon, **kwargs):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_values = scaler.fit_transform(train_values)
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = scaled_train_values[:, region]
        try:
            arima_model = ARIMA(region_data, order=(5, 1, 0))
            arima_fitted_model = arima_model.fit(disp=False)
            forecast = arima_fitted_model.forecast(steps=horizon)
        except Exception as e:
            forecast = np.full(horizon, np.nan)
        forecast = np.maximum(forecast, 0)
        inverse_transformed_forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
        inverse_transformed_forecast[np.isnan(inverse_transformed_forecast)] = 0
        forecasted_values[:, region] = inverse_transformed_forecast
    return forecasted_values