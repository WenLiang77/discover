import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

def decompose_and_forecast(region_data, horizon, seasonal_period):
    decomposition = seasonal_decompose(region_data, model='multiplicative', period=seasonal_period)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    try:
        arima_trend = ARIMA(trend, order=(1, 1, 1)).fit(disp=False)
        trend_forecast = arima_trend.forecast(steps=horizon)
    except Exception as e:
        trend_forecast = np.nanmean(trend[-horizon:], axis=0)
    try:
        sarimax_seasonal = SARIMAX(seasonal, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period)).fit(disp=False)
        seasonal_forecast = sarimax_seasonal.forecast(steps=horizon)
    except Exception as e:
        seasonal_forecast = np.nanmean(seasonal[-horizon:], axis=0)
    try:
        exponential_residual = ExponentialSmoothing(residual, trend=None, seasonal='add', seasonal_periods=seasonal_period).fit(disp=False)
        residual_forecast = exponential_residual.forecast(steps=horizon)
    except Exception as e:
        residual_forecast = np.nanmean(residual[-horizon:], axis=0)
    forecast = trend_forecast * seasonal_forecast * residual_forecast
    forecast = np.clip(forecast, 0, None)
    return forecast

def covid_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecasts = np.zeros((horizon, n_regions))
    for i in range(n_regions):
        region_data = train_values[:, i]
        try:
            seasonal_period = min(len(region_data) // 2, 365)
            forecast = decompose_and_forecast(region_data, horizon, seasonal_period)
        except Exception as e:
            forecast = np.nanmean(region_data[-horizon:], axis=0)
        forecasts[:, i] = forecast
    return forecasts