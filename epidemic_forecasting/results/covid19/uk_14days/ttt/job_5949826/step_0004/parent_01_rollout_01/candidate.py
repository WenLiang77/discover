import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    imputer = SimpleImputer(strategy='mean')
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data = imputer.fit_transform(region_data.reshape(-1, 1)).flatten()
        seasonal_period = 7
        if len(region_data) >= 2 * seasonal_period:
            decomposition = seasonal_decompose(region_data, model='multiplicative', period=seasonal_period)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
        else:
            trend = np.zeros_like(region_data)
            seasonal = np.ones_like(region_data)
        combined_data = trend * seasonal
        try:
            sarimax_model = SARIMAX(combined_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))
            sarimax_fitted_model = sarimax_model.fit(disp=False)
            trend_forecast = sarimax_fitted_model.get_forecast(steps=horizon).predicted_mean
            seasonal_forecast = seasonal[:horizon]
            combined_forecast = trend_forecast * seasonal_forecast
        except Exception as _:
            combined_forecast = np.full(horizon, np.nan)
        try:
            es_model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
            es_fitted_model = es_model.fit()
            forecast = es_fitted_model.forecast(steps=horizon)
        except Exception as _:
            forecast = np.full(horizon, np.nan)
        combined_forecast[np.isnan(combined_forecast)] = forecast[np.isnan(combined_forecast)]
        forecasted_values[:, region] = np.maximum(combined_forecast, 0)
    return forecasted_values