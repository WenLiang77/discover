import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import smape, mae, rmse, mase

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        try:
            model = SARIMAX(train_values[:, region], order=(2, 0, 1), seasonal_order=(1, 0, 1, 52))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=horizon)
            forecasted_values[:, region] = forecast
        except Exception as e:
            window_size = min(10, len(train_values[:, region]))
            smoothed_values = np.convolve(train_values[:, region], np.ones(window_size) / window_size, mode='valid')
            forecasted_values[:len(smoothed_values), region] = smoothed_values
            forecasted_values[len(smoothed_values):, region] = np.mean(train_values[:, region])
    forecasted_values = np.maximum(forecasted_values, 0)
    return forecasted_values