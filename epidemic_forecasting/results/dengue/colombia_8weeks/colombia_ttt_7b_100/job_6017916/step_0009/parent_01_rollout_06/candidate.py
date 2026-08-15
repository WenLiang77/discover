import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions), dtype=float)
    for region in range(num_regions):
        region_data = train_values[:, region]
        if is_stationary(region_data):
            try:
                model = SARIMAX(region_data, order=(1, 0, 1), seasonal_order=(1, 0, 1, 52)).fit(disp=False)
            except Exception as e:
                model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False)
        else:
            try:
                model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52)).fit(disp=False)
            except Exception as e:
                model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False)
        forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
        forecasted_values[:, region] = forecasted_region_data.clip(0)
    return forecasted_values

def is_stationary(time_series, alpha=0.05):
    n = len(time_series)
    mean = np.mean(time_series)
    variance = np.var(time_series)
    z_scores = [(x - mean) / np.sqrt(variance) for x in time_series]
    for i in range(n - 1):
        z_diff = abs(z_scores[i] - z_scores[i + 1])
        if z_diff > 1.96:
            return False
    return True