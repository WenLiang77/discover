import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions), dtype=float)
    for region in range(num_regions):
        region_data = train_values[:, region]
        if len(region_data) < 2:
            forecasted_values[:, region] = np.mean(region_data)
            continue
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
        except Exception as e:
            forecasted_values[:, region] = np.mean(region_data)
            continue
        forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
        forecasted_values[:, region] = forecasted_region_data
    return forecasted_values.clip(min=0)