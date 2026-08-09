import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

def preprocess_data(data):
    data[data == 0] = 1e-06
    log_data = np.log(data)
    decomposition = seasonal_decompose(log_data, model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    trend.fillna(method='ffill', inplace=True)
    trend.fillna(method='bfill', inplace=True)
    seasonal.fillna(seasonal.mean(), inplace=True)
    preprocessed_data = np.exp(trend + seasonal)
    return preprocessed_data

def fit_and_forecast(model_class, train_data, horizon):
    try:
        model = model_class(train_data, freq='D')
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=horizon)
        return forecast.clip(0)
    except Exception as e:
        return np.full(horizon, np.nan)

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        preprocessed_data = preprocess_data(region_data)
        sarimax_forecast = fit_and_forecast(SARIMAX, preprocessed_data, horizon)
        if ~np.isnan(sarimax_forecast).all():
            forecasted_values[:, region] = sarimax_forecast
            continue
        exponential_forecast = fit_and_forecast(ExponentialSmoothing, preprocessed_data, horizon)
        if ~np.isnan(exponential_forecast).all():
            forecasted_values[:, region] = exponential_forecast
            continue
    forecasted_values[np.isnan(forecasted_values)] = 0
    return forecasted_values