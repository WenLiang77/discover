import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        try:
            model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit()
        except Exception as e:
            forecasted_values[:, region] = np.mean(region_data)
            continue
        forecasted_region_data = model.forecast(steps=horizon)
        forecasted_values[:, region] = forecasted_region_data
    return forecasted_values