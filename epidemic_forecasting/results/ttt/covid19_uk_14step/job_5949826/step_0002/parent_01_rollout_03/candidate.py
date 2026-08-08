import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        decomposition = seasonal_decompose(region_data, model='additive')
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        scaler = StandardScaler()
        scaled_residuals = scaler.fit_transform(residual.reshape(-1, 1)).flatten()
        combined_series = trend + seasonal + scaled_residuals
        try:
            model = ARIMA(combined_series, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=horizon)
            inverse_forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
            inverse_forecast += trend[-horizon:] - trend[-horizon - 1:-1]
            inverse_forecast += seasonal[-horizon:] - seasonal[-horizon - 1:-1]
            forecasted_values[:, region] = np.maximum(inverse_forecast, 0)
        except Exception as e:
            forecasted_values[:, region] = 0
    return forecasted_values