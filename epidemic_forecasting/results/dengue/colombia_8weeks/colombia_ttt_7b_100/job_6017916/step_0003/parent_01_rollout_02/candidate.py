import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if np.allclose(region_data, region_data[0], atol=1e-06):
            forecasted_values[:, region] = region_data[-1]
            continue
        train_size = int(len(region_data) * 0.8)
        train_data = region_data[:train_size]
        val_data = region_data[train_size:]
        sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
        forecasted_region_data = sarima_model.get_forecast(steps=horizon).predicted_mean
        forecasted_region_data = np.maximum(forecasted_region_data, 0)
        forecasted_values[:, region] = forecasted_region_data
    return forecasted_values