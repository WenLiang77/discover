import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def fit_exponential_smoothing(data, trend='add', seasonal=None, seasonal_periods=7):
    try:
        model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        fitted_model = model.fit(disp=False)
        return fitted_model
    except Exception as e:
        print(f'Exponential Smoothing fit failed: {e}')
        return None

def fit_sarimax(data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7)):
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        return fitted_model
    except Exception as e:
        print(f'SARIMAX fit failed: {e}')
        return None

def covid_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    preprocessed_train_values = preprocess_data(train_values)
    for region in range(num_regions):
        region_data = preprocessed_train_values[:, region].reshape(-1, 1)
        es_model = fit_exponential_smoothing(region_data.flatten(), trend='add', seasonal='mul', seasonal_periods=7)
        if es_model is not None:
            forecast_es = es_model.forecast(steps=horizon).reshape(-1, 1)
            forecasted_values[:, region] = np.clip(np.exp(forecast_es), 0, None)
        sarimax_model = fit_sarimax(region_data.flatten())
        if sarimax_model is not None:
            forecast_sarimax = sarimax_model.forecast(steps=horizon).reshape(-1, 1)
            forecasted_values[:, region] = np.clip(forecast_sarimax, 0, None)
    return forecasted_values