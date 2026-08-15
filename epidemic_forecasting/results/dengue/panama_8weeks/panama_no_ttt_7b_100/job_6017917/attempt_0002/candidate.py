import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

def dengue_forecast(train_values, horizon, **kwargs):
    train_values = np.array(train_values)
    N = train_values.shape[1]
    forecasts = []
    for i in range(N):
        region_series = train_values[:, i]
        imputer = SimpleImputer(strategy='mean')
        region_series_imputed = imputer.fit_transform(region_series.reshape(-1, 1)).flatten()
        pipeline = make_pipeline(StandardScaler(), SARIMAX(order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)))
        try:
            model = pipeline.fit(region_series_imputed)
            forecast = model.get_forecast(steps=horizon).predicted_mean
            forecast = np.maximum(forecast, 0)
        except Exception as e:
            forecast = np.convolve(region_series_imputed, np.ones(horizon) / horizon, mode='valid')
            forecast = np.concatenate([np.zeros(horizon - len(forecast)), forecast])
            forecast = np.maximum(forecast, 0)
        forecasts.append(forecast)
    forecasts_array = np.array(forecasts).reshape(horizon, N)
    return forecasts_array