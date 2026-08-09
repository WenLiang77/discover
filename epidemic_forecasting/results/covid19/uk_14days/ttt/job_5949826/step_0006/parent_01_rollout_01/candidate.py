import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import PowerTransformer

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        if len(region_data) < 2:
            continue
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
        try:
            decomposition = seasonal_decompose(transformed_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            sarimax_model_trend = SARIMAX(trend, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted_model_trend = sarimax_model_trend.fit(disp=False)
            es_model_seasonal = ExponentialSmoothing(seasonal, trend='add', seasonal_periods=7)
            es_fitted_model_seasonal = es_model_seasonal.fit()
            trend_forecast = sarimax_fitted_model_trend.get_forecast(steps=horizon).predicted_mean
            seasonal_forecast = es_fitted_model_seasonal.forecast(steps=horizon)
            combined_forecast = trend_forecast + seasonal_forecast
            inverse_transformed_forecast = pt.inverse_transform(combined_forecast.reshape(-1, 1)).flatten()
            forecasted_values[:, region] = np.maximum(inverse_transformed_forecast, 0)
        except Exception as e:
            try:
                es_model_direct = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                es_fitted_model_direct = es_model_direct.fit()
                forecast = es_fitted_model_direct.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values