import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        imputer = SimpleImputer(strategy='mean')
        region_data_imputed = imputer.fit_transform(region_data.reshape(-1, 1)).flatten()
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(region_data_imputed.reshape(-1, 1)).flatten()
        try:
            decomposition = seasonal_decompose(transformed_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            arima_trend_model = SARIMAX(trend, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            arima_trend_fitted_model = arima_trend_model.fit(disp=False)
            trend_forecast = arima_trend_fitted_model.get_forecast(steps=horizon).predicted_mean
            es_seasonal_model = ExponentialSmoothing(seasonal, trend='add', seasonal_periods=7)
            es_seasonal_fitted_model = es_seasonal_model.fit()
            seasonal_forecast = es_seasonal_fitted_model.forecast(steps=horizon)
            combined_forecast = trend_forecast + seasonal_forecast
            inverse_transformed_forecast = pt.inverse_transform(combined_forecast.reshape(-1, 1)).flatten()
            forecasted_values[:, region] = np.maximum(inverse_transformed_forecast, 0)
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(region_data_imputed, trend='add', seasonal=None)
                es_fitted_model = es_model.fit()
                forecast = es_fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values