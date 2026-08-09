import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        scaled_data = scaler.fit_transform(region_data.reshape(-1, 1))
        try:
            decomposition = seasonal_decompose(scaled_data.flatten(), model='additive', period=7)
            sarimax_model = SARIMAX(decomposition.resid, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted = sarimax_model.fit(disp=False)
            sarimax_forecast = sarimax_fitted.get_forecast(steps=horizon)
            sarimax_forecast_values = sarimax_forecast.predicted_mean
            trend_forecast = decomposition.trend[-1] + (decomposition.trend[-1] - decomposition.trend[-2]) * np.arange(1, horizon + 1)
            seasonal_forecast = decomposition.seasonal[-7:] + (decomposition.seasonal[-7:] - decomposition.seasonal[-8:-1]) * np.arange(1, horizon + 1)
            forecast = sarimax_forecast_values * decomposition.seasonal[-1] + trend_forecast
            forecasted_values[:, region] = scaler.inverse_transform(forecast.reshape(-1, 1))[:, 0].clip(min=0)
        except Exception as _:
            try:
                es_model = ExponentialSmoothing(scaled_data.flatten(), trend='add', seasonal=None)
                es_fitted = es_model.fit()
                es_forecast = es_fitted.forecast(steps=horizon)
                forecasted_values[:, region] = scaler.inverse_transform(es_forecast.reshape(-1, 1))[:, 0].clip(min=0)
            except Exception as _:
                forecasted_values[:, region] = 0
    return forecasted_values