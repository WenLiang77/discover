import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_values = scaler.fit_transform(train_values)
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = scaled_train_values[:, region]
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
            forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
        except Exception as _:
            try:
                model = ARIMA(region_data, order=(1, 1, 1)).fit(disp=False)
                forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
            except Exception as _:
                forecasted_region_data = np.full(horizon, np.nanmean(region_data))
        forecasted_region_data = np.maximum(forecasted_region_data, 0)
        forecasted_region_data = scaler.inverse_transform(np.array([forecasted_region_data]).T)[:, 0]
        forecasted_values[:, region] = forecasted_region_data
    return forecasted_values