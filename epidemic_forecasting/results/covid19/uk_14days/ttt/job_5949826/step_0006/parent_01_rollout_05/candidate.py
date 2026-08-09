import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from sklearn.exceptions import NotFittedError

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region].astype(np.float64)
        if len(region_data) < horizon * 2:
            continue
        try:
            pt = PowerTransformer(method='yeo-johnson')
            transformed_data = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
            decomposition = seasonal_decompose(transformed_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            sarimax_model_trend = SARIMAX(trend, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted_model_trend = sarimax_model_trend.fit(disp=False)
            trend_forecast = sarimax_fitted_model_trend.get_forecast(steps=horizon).predicted_mean
            sarimax_model_seasonal = SARIMAX(seasonal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted_model_seasonal = sarimax_model_seasonal.fit(disp=False)
            seasonal_forecast = sarimax_fitted_model_seasonal.get_forecast(steps=horizon).predicted_mean
            combined_forecast = trend_forecast + seasonal_forecast
            inverse_transformed_forecast = pt.inverse_transform(combined_forecast.reshape(-1, 1)).flatten()
            forecasted_values[:, region] = np.maximum(inverse_transformed_forecast, 0)
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                es_fitted_model = es_model.fit(disp=False)
                forecasted_values[:, region] = np.maximum(es_fitted_model.forecast(steps=horizon), 0)
            except Exception as e:
                pass
    return forecasted_values