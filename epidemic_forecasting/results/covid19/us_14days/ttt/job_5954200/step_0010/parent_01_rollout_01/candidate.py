import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

def covid_forecast(train_values, horizon, **kwargs):
    forecasted_values = np.zeros((horizon, train_values.shape[1]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_data += 1e-06
        region_data_scaled = scaler.fit_transform(region_data.reshape(-1, 1)).flatten()
        best_model = None
        best_performance = float('inf')
        models = [{'model': SARIMAX, 'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 7)}, {'model': ExponentialSmoothing, 'trend': 'add', 'seasonal': None}]
        for model_info in models:
            try:
                if model_info['model'] == SARIMAX:
                    model = model_info['model'](region_data_scaled, order=model_info['order'], seasonal_order=model_info['seasonal_order'])
                else:
                    model = model_info['model'](region_data_scaled, trend=model_info['trend'], seasonal=model_info['seasonal'])
                fitted_model = model.fit(disp=False)
                forecast_scaled = fitted_model.get_forecast(steps=horizon).predicted_mean
                mse = np.mean((forecast_scaled - region_data_scaled[-horizon:]) ** 2)
                if mse < best_performance:
                    best_model = fitted_model
                    best_performance = mse
            except (ValueError, AttributeError):
                continue
        if best_model is None:
            best_model = ExponentialSmoothing(region_data_scaled, trend='add', seasonal=None)
            best_model.fit()
        forecast_scaled = best_model.get_forecast(steps=horizon).predicted_mean
        forecast_original_scale = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        forecasted_values[:, region] = np.maximum(forecast_original_scale, 0)
    return forecasted_values