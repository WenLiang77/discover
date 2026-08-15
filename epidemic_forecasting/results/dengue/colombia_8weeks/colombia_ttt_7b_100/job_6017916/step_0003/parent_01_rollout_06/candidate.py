import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions), dtype=float)
    for region in range(num_regions):
        region_data = train_values[:, region]
        diff_order = 0
        if np.diff(region_data).var() > 0.01 * region_data.var():
            diff_order = 1
        try:
            model = SARIMAX(region_data, order=(1, diff_order, 1), seasonal_order=(1, 0, 1, 52)).fit(disp=False)
        except Exception as e:
            model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False)
        forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
        forecasted_values[:, region] = forecasted_region_data.clip(0)
    return forecasted_values