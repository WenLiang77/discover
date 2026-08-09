import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer

def preprocess_data(train_values):
    imputer = SimpleImputer(strategy='mean')
    transformed_values = imputer.fit_transform(train_values)
    scaler = PowerTransformer(method='yeo-johnson')
    scaled_values = scaler.fit_transform(transformed_values)
    decomposed_values = []
    for region_data in scaled_values.T:
        decomposition = seasonal_decompose(region_data, model='multiplicative', period=7)
        seasonal_data = decomposition.seasonal
        decomposed_values.append(seasonal_data)
    return np.array(decomposed_values).T

def fit_models(train_values, horizon):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=horizon).predicted_mean
            forecasted_values[:, region] = np.maximum(forecast, 0)
        except Exception as _:
            pass
        if np.allclose(forecasted_values[:, region], 0):
            try:
                model = ARIMA(region_data, order=(1, 1, 1))
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as _:
                pass
        if np.allclose(forecasted_values[:, region], 0):
            try:
                model = ExponentialSmoothing(region_data, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=horizon)
                forecasted_values[:, region] = np.maximum(forecast, 0)
            except Exception as _:
                pass
    return forecasted_values

def covid_forecast(train_values, horizon, **kwargs):
    preprocessed_values = preprocess_data(train_values)
    forecasted_values = fit_models(preprocessed_values, horizon)
    return forecasted_values