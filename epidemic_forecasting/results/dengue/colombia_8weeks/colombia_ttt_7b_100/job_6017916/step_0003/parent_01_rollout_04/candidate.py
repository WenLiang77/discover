import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions), dtype=float)
    for region in range(num_regions):
        region_data = train_values[:, region]
        if len(region_data) < 2:
            forecasted_values[:, region] = np.mean(region_data)
            continue
        try:
            model_arima = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
            forecast_arima = model_arima.get_forecast(steps=horizon)
            model_es = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False)
            forecast_es = model_es.get_forecast(steps=horizon)
            weights = [0.6, 0.4]
            combined_forecast = weights[0] * forecast_arima.predicted_mean + weights[1] * forecast_es.predicted_mean
            forecasted_values[:, region] = np.maximum(combined_forecast, 0)
        except Exception as e:
            forecasted_values[:, region] = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False).forecast(steps=horizon)
    return forecasted_values