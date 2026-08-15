import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import smape, mae, rmse, mase

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if np.all(region_data == 0):
            forecasted_values[:, region] = 0
            continue
        try:
            model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
        except Exception as e:
            model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False)
        forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
        forecasted_values[:, region] = forecasted_region_data
    forecasted_values = np.maximum(forecasted_values, 0)
    return forecasted_values