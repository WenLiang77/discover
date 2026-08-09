import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        imputer = SimpleImputer(strategy='mean')
        region_data = imputer.fit_transform(region_data.reshape(-1, 1)).flatten()
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
        try:
            decomposition = seasonal_decompose(transformed_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            sarimax_model_trend = SARIMAX(trend, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted_model_trend = sarimax_model_trend.fit(disp=False)
            trend_forecast = sarimax_fitted_model_trend.get_forecast(steps=horizon).predicted_mean
            arima_model_seasonal = ARIMA(seasonal, order=(1, 1, 1))
            arima_fitted_model_seasonal = arima_model_seasonal.fit(disp=False)
            seasonal_forecast = arima_fitted_model_seasonal.forecast(steps=horizon)
            combined_forecast = trend_forecast + seasonal_forecast
            inverse_transformed_forecast = pt.inverse_transform(combined_forecast.reshape(-1, 1)).flatten()
            forecasted_values[:, region] = np.maximum(inverse_transformed_forecast, 0)
        except Exception as e:
            try:
                sarimax_model_direct = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                sarimax_fitted_model_direct = sarimax_model_direct.fit(disp=False)
                direct_forecast = sarimax_fitted_model_direct.get_forecast(steps=horizon).predicted_mean
                forecasted_values[:, region] = np.maximum(direct_forecast, 0)
            except Exception as e:
                es_model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                es_fitted_model = es_model.fit()
                fallback_forecast = es_fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(fallback_forecast, 0)
    return forecasted_values