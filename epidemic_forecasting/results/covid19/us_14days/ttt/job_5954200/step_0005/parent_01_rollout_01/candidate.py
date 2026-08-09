import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def preprocess_data(data):
    data_log = np.log1p(data)
    decomposition = seasonal_decompose(data_log, model='multiplicative', period=7)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    trend.fillna(0, inplace=True)
    seasonal.fillna(0, inplace=True)
    return (data_log, trend, seasonal)

def fit_exponential_smoothing(data, initial_level=None, trend=None):
    model = ExponentialSmoothing(data, initialization_method='estimated', trend=trend)
    fitted_model = model.fit(start_params=[initial_level])
    return fitted_model

def fit_sarimax(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        return fitted_model
    except Exception as _:
        return None

def fit_linear_regression(data):
    X = np.arange(len(data)).reshape(-1, 1)
    model = LinearRegression().fit(X, data)
    return model

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        data_log, trend, seasonal = preprocess_data(region_data)
        es_model = fit_exponential_smoothing(data_log, initial_level=data_log.mean(), trend='add')
        if es_model:
            forecast_es = es_model.forecast(horizon)
            forecast_es = np.maximum(np.exp(forecast_es) - 1, 0)
        else:
            forecast_es = np.zeros(horizon)
        sarimax_model = fit_sarimax(data_log)
        if sarimax_model:
            forecast_sarimax = sarimax_model.forecast(horizon)
            forecast_sarimax = np.maximum(np.exp(forecast_sarimax) - 1, 0)
        else:
            forecast_sarimax = np.zeros(horizon)
        linreg_trend = fit_linear_regression(trend)
        linreg_seasonal = fit_linear_regression(seasonal)
        if linreg_trend and linreg_seasonal:
            future_trend = linreg_trend.predict(np.arange(len(region_data), len(region_data) + horizon))
            future_seasonal = linreg_seasonal.predict(np.arange(len(region_data), len(region_data) + horizon))
            forecast_linreg = (np.exp(future_trend + future_seasonal) - 1) * data_log.std() + data_log.mean()
            forecast_linreg = np.maximum(forecast_linreg, 0)
        else:
            forecast_linreg = np.zeros(horizon)
        forecasted_values[:, region] = np.clip((forecast_es + forecast_sarimax + forecast_linreg) / 3, 0, None)
    return forecasted_values