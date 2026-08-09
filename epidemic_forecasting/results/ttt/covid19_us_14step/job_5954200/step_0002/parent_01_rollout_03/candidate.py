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
        combined_data = np.column_stack((trend, seasonal, scaled_residuals))
        try:
            model = SARIMAX(combined_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fitted_model = model.fit(disp=False)
            forecast_combined = fitted_model.get_forecast(steps=horizon).forecast_components
            forecast_scaled_residuals = scaler.inverse_transform(forecast_combined[:, -1].reshape(-1, 1))
            forecast_trend = forecast_combined[:, 0]
            forecast_seasonal = forecast_combined[:, 1]
            forecast_residuals = forecast_scaled_residuals.flatten()
            forecast_values = forecast_trend + forecast_seasonal + forecast_residuals
            forecast_values = np.maximum(forecast_values, 0)
            forecasted_values[:, region] = forecast_values
        except Exception as e:
            forecasted_values[:, region] = 0
    return forecasted_values