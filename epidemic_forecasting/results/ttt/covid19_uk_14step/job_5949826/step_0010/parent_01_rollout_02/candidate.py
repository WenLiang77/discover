import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from math import isnan, inf

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]), dtype=float)
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        if np.all(region_data == 0):
            continue
        log_data = np.where(region_data == 0, 1e-06, region_data)
        log_data = np.log(log_data)
        try:
            decomposition = seasonal_decompose(log_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            if trend is None or seasonal is None:
                raise ValueError('Decomposition failed')
            sarima_model = SARIMAX(trend, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarima_fitted_model = sarima_model.fit(disp=False)
            trend_forecast = sarima_fitted_model.get_forecast(steps=horizon).predicted_mean
            es_model = ExponentialSmoothing(seasonal, trend='add', seasonal_periods=7)
            es_fitted_model = es_model.fit()
            seasonal_forecast = es_fitted_model.forecast(steps=horizon)
            combined_forecast = np.exp(trend_forecast + seasonal_forecast) - 1
            forecasted_values[:, region] = np.clip(combined_forecast, 0, None)
        except Exception as e:
            try:
                es_model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                es_fitted_model = es_model.fit()
                forecast = es_fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.clip(forecast, 0, None)
            except Exception as e:
                forecasted_values[:, region] = 0
    return forecasted_values