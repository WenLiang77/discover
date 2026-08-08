import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox
from warnings import filterwarnings

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1] < 0.05

def select_arima_order(timeseries):
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    best_aic = float('inf')
    best_order = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    model = ARIMA(timeseries, order=order)
                    model_fit = model.fit(disp=False)
                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                except:
                    continue
    return best_order

def select_sarimax_order(timeseries, seasonal_period):
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    P_values = range(0, 2)
    D_values = range(0, 1)
    Q_values = range(0, 2)
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, seasonal_period)
                            try:
                                model = SARIMAX(timeseries, order=order, seasonal_order=seasonal_order)
                                model_fit = model.fit(disp=False)
                                aic = model_fit.aic
                                if aic < best_aic:
                                    best_aic = aic
                                    best_order = order
                                    best_seasonal_order = seasonal_order
                            except:
                                continue
    return (best_order, best_seasonal_order)

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        if not check_stationarity(region_data):
            try:
                _, lambda_value = boxcox(region_data)
                region_data = boxcox(region_data, lmbda=lambda_value)
            except:
                pass
        seasonal_period = kwargs.get('frequency', 7)
        if seasonal_period > 1:
            arima_order, sarimax_order = select_sarimax_order(region_data, seasonal_period)
            model = SARIMAX(region_data, order=sarimax_order, seasonal_order=sarimax_order)
        else:
            arima_order = select_arima_order(region_data)
            model = ARIMA(region_data, order=arima_order)
        try:
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            forecasted_values[:, region] = np.maximum(forecast, 0)
        except Exception as e:
            print(f'Error in fitting model for region {region}: {e}')
    return forecasted_values