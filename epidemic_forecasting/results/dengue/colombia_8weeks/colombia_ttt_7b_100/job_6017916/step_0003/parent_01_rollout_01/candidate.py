import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if np.all(region_data == 0):
            forecasted_values[:, region] = 0
            continue
        if np.abs(np.diff(region_data)).mean() > 0:
            try:
                model = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
            except Exception as e:
                model = ARIMA(region_data, order=(1, 1, 1)).fit(disp=False)
        else:
            model = ARIMA(region_data, order=(1, 0, 1)).fit(disp=False)
        forecasted_region_data = model.forecast(steps=horizon)
        forecasted_values[:, region] = forecasted_region_data
    forecasted_values = np.maximum(forecasted_values, 0)
    return forecasted_values