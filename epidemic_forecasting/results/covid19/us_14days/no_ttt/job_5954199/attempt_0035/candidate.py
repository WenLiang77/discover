import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

def preprocess_data(train_values):
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    scaled_values = scaler.fit_transform(imputer.fit_transform(train_values))
    return (scaled_values, scaler, imputer)

def fit_models(scaled_values, horizon):
    models = []
    for i in range(scaled_values.shape[1]):
        try:
            arima_model = ARIMA(scaled_values[:, i], order=(5, 1, 0)).fit(disp=False)
            models.append(arima_model)
        except Exception:
            expsmo_model = ExponentialSmoothing(scaled_values[:, i]).fit(smoothing_level=0.2, optimized=False)
            models.append(expsmo_model)
    forecasts = np.zeros((horizon, scaled_values.shape[1]))
    for i, model in enumerate(models):
        forecasted_values = model.forecast(steps=horizon)
        forecasts[:, i] = forecasted_values
    return forecasts

def postprocess_forecasts(forecasts, scaler, imputer):
    inverse_transformed = scaler.inverse_transform(forecasts)
    imputed_inverse = imputer.inverse_transform(inverse_transformed)
    clipped_forecasts = np.clip(imputed_inverse, 0, None)
    return clipped_forecasts

def covid_forecast(train_values, horizon, **kwargs):
    scaled_values, scaler, imputer = preprocess_data(train_values)
    forecasts = fit_models(scaled_values, horizon)
    clipped_forecasts = postprocess_forecasts(forecasts, scaler, imputer)
    return clipped_forecasts