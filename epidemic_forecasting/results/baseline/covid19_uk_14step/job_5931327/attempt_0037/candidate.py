import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error
from typing import Optional

def preprocess_data(train_values):
    train_values = np.where(train_values == 0, 1e-06, train_values)
    return np.log(train_values)

def postprocess_data(forecasted_values):
    forecasted_values = np.exp(forecasted_values)
    forecasted_values = np.maximum(forecasted_values, 0)
    return forecasted_values

def covid_forecast(train_values: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
    N = train_values.shape[1]
    forecasted_values = np.zeros((horizon, N))
    for region in range(N):
        try:
            processed_train = preprocess_data(train_values[:, region])
            model = SARIMAX(processed_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            model_fit = model.fit(disp=False)
            forecasted_processed = model_fit.forecast(steps=horizon)
            forecasted_values[:, region] = postprocess_data(forecasted_processed)
        except Exception as e:
            pass
    return forecasted_values