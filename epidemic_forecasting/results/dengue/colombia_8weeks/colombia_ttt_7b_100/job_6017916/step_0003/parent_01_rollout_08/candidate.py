import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

def stationarity_test(data):
    result = adfuller(data)
    return result[1] < 0.05

def difference_series(data):
    return data.diff().dropna()

def decompose_series(data):
    decomposition = seasonal_decompose(data, model='additive')
    return (decomposition.trend, decomposition.seasonal, decomposition.resid)

def fit_sarimax(data, order, seasonal_order):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)

def forecast_sarimax(model, horizon):
    return model.forecast(steps=horizon)

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if not stationarity_test(region_data):
            diff_data = difference_series(region_data)
        else:
            diff_data = region_data.copy()
        try:
            best_model = auto_arima(diff_data, start_p=0, start_q=0, test='adf', max_p=3, max_q=3, m=52, seasonal=True, stepwise=True, suppress_warnings=True)
            order, seasonal_order = (best_model.order, best_model.seasonal_order)
            sarimax_model = fit_sarimax(diff_data, order, seasonal_order)
            forecast_diff = forecast_sarimax(sarimax_model, horizon)
            forecast_actual = np.cumsum(forecast_diff) + region_data[-1]
            forecast_actual = np.maximum(forecast_actual, 0)
            forecasted_values[:, region] = forecast_actual
        except Exception as e:
            forecasted_values[:, region] = np.full(horizon, np.mean(region_data))
    return forecasted_values