import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(region_data.reshape(-1, 1)).flatten()
        try:
            decomposition = seasonal_decompose(transformed_data, model='additive', period=7)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            arima_model = ARIMA(trend, order=(1, 1, 1))
            arima_fitted_model = arima_model.fit(disp=False)
            trend_forecast = arima_fitted_model.predict(start=len(trend), end=len(trend) + horizon - 1)
            sarimax_model = SARIMAX(seasonal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarimax_fitted_model = sarimax_model.fit(disp=False)
            seasonal_forecast = sarimax_fitted_model.predict(start=len(seasonal), end=len(seasonal) + horizon - 1)
            combined_forecast = trend_forecast + seasonal_forecast
            inverse_transformed_forecast = pt.inverse_transform(combined_forecast.reshape(-1, 1)).flatten()
            forecasted_values[:, region] = np.maximum(inverse_transformed_forecast, 0)
        except Exception as e:
            try:
                arima_model = ARIMA(region_data, order=(1, 1, 1))
                arima_fitted_model = arima_model.fit(disp=False)
                forecast = arima_fitted_model.predict(start=len(region_data), end=len(region_data) + horizon - 1)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as e:
                model = LinearRegression()
                model.fit(np.arange(len(region_data)).reshape(-1, 1), region_data)
                forecast = model.predict(np.arange(len(region_data), len(region_data) + horizon).reshape(-1, 1))
                forecasted_values[:, region] = np.maximum(forecast, 0)
    return forecasted_values