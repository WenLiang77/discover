import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        try:
            decomposition = seasonal_decompose(region_data, model='multiplicative', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            arima_model = ARIMA(residual, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            residual_forecast = arima_fit.forecast(steps=horizon)
            forecast_trend = np.repeat(trend[-1], horizon)
            forecast_seasonal = np.repeat(seasonal[-1], horizon)
            forecast = forecast_trend + forecast_seasonal + residual_forecast
            smooth_model = ExponentialSmoothing(forecast, trend='add', seasonal=None)
            smooth_fit = smooth_model.fit()
            final_forecast = smooth_fit.forecast(steps=horizon)
            forecasted_values[:, region] = np.maximum(final_forecast, 0)
        except Exception as _:
            try:
                sarimax_model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                sarimax_fit = sarimax_model.fit(disp=False)
                forecast = sarimax_fit.get_forecast(steps=horizon).predicted_mean
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as _:
                try:
                    es_model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                    es_fit = es_model.fit()
                    forecast = es_fit.forecast(steps=horizon)
                    forecasted_values[:, region] = np.maximum(forecast, 0)
                except Exception as _:
                    pass
    return forecasted_values