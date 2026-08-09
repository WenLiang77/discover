import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    imputer = SimpleImputer(strategy='mean')
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data_imputed = imputer.fit_transform(region_data.reshape(-1, 1)).flatten()
        try:
            decomposition = seasonal_decompose(region_data_imputed, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            sarimax_model = SARIMAX(trend, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted_model = sarimax_model.fit(disp=False)
            trend_forecast = sarimax_fitted_model.get_forecast(steps=horizon).predicted_mean
            es_model = ExponentialSmoothing(seasonal, trend='add', seasonal_periods=7)
            es_fitted_model = es_model.fit()
            seasonal_forecast = es_fitted_model.forecast(steps=horizon)
            combined_forecast = trend_forecast + seasonal_forecast
            forecasted_values[:, region] = np.maximum(combined_forecast, 0)
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(region_data_imputed, trend='add', seasonal=None)
                es_fitted_model = es_model.fit()
                forecast = es_fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as e:
                pass
    return forecasted_values