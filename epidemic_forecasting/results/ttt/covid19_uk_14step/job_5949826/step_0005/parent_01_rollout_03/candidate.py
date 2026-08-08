import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import PowerTransformer
from scipy.stats.mstats import gmean

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
        try:
            sarimax_model = SARIMAX(transformed_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted_model = sarimax_model.fit(maxiter=1000, disp=False)
            sarimax_forecast = sarimax_fitted_model.get_forecast(steps=horizon).predicted_mean
            forecasted_values[:, region] = np.maximum(pt.inverse_transform(sarimax_forecast.reshape(-1, 1)), 0)
        except Exception:
            try:
                arima_model = ARIMA(transformed_data, order=(1, 1, 1))
                arima_fitted_model = arima_model.fit(maxiter=1000, disp=False)
                arima_forecast = arima_fitted_model.get_forecast(steps=horizon).predicted_mean
                forecasted_values[:, region] = np.maximum(pt.inverse_transform(arima_forecast.reshape(-1, 1)), 0)
            except Exception:
                try:
                    es_model = ExponentialSmoothing(transformed_data, trend='add', seasonal_periods=7)
                    es_fitted_model = es_model.fit(smoothing_level=0.2, smoothing_slope=0.2, smoothing_seasonal=0.2, optimized=True)
                    es_forecast = es_fitted_model.forecast(steps=horizon)
                    forecasted_values[:, region] = np.maximum(pt.inverse_transform(es_forecast.reshape(-1, 1)), 0)
                except Exception:
                    es_fallback_model = ExponentialSmoothing(transformed_data, trend=None, seasonal=None)
                    es_fallback_fitted_model = es_fallback_model.fit(smoothing_level=0.2, optimized=True)
                    es_fallback_forecast = es_fallback_fitted_model.forecast(steps=horizon)
                    forecasted_values[:, region] = np.maximum(pt.inverse_transform(es_fallback_forecast.reshape(-1, 1)), 0)
    return forecasted_values