import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.impute import SimpleImputer

def dengue_forecast(train_values, horizon, **kwargs):
    imputer = SimpleImputer(strategy='mean')
    train_values_imputed = imputer.fit_transform(train_values)
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        try:
            model = SARIMAX(train_values_imputed[:, i], order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
            results = model.fit(disp=False)
            forecast[:, i] = results.get_forecast(steps=horizon).predicted_mean
        except Exception as e:
            ewm_value = train_values_imputed[-1, i]
            forecast[:, i] = ewm_value * (np.arange(1, horizon + 1) / np.arange(1, horizon + 1)).sum()
    forecast = np.maximum(forecast, 0)
    return forecast