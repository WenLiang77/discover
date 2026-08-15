import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if len(region_data) < 2 * horizon:
            forecasted_values[:, region] = np.mean(region_data)
            continue
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
            forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
        except Exception as e:
            model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit()
            forecasted_region_data = model.forecast(steps=horizon)
        forecasted_region_data = np.maximum(forecasted_region_data, 0)
        forecasted_values[:, region] = forecasted_region_data
    return forecasted_values