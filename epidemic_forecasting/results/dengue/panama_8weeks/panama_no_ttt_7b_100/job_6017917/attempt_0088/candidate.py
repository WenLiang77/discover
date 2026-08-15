import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

def preprocess_data(train_values):
    scaler = StandardScaler()
    return scaler.fit_transform(train_values)

def fit_sarimax_model(data, order=(1, 1, 0), seasonal_order=(1, 1, 0, 52)):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)

def forecast_with_sarimax(model, horizon):
    forecasted = model.get_forecast(steps=horizon).predicted_mean
    return forecasted

def dengue_forecast(train_values, horizon, **kwargs):
    scaled_train = preprocess_data(train_values)
    forecasts = []
    for region in range(scaled_train.shape[1]):
        region_data = scaled_train[:, region]
        try:
            sarimax_model = fit_sarimax_model(region_data)
            region_forecast = forecast_with_sarimax(sarimax_model, horizon)
            forecasts.append(region_forecast)
        except Exception as e:
            print(f'Error fitting SARIMAX model for region {region}: {e}')
            forecasts.append(np.zeros(horizon))
    forecast_array = np.column_stack(forecasts)
    forecast_array = train_values[-1].mean() + train_values.std() * forecast_array
    forecast_array = np.clip(forecast_array, 0, None)
    return forecast_array