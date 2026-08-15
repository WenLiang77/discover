import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if np.allclose(region_data, region_data[0], atol=1e-05):
            forecasted_values[:, region] = region_data[0]
            continue
        try:
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 52)
            model = SARIMAX(region_data, order=order, seasonal_order=seasonal_order).fit(disp=False)
        except Exception as e:
            print(f'Failed to fit SARIMAX model for region {region}: {e}')
            forecasted_values[:, region] = np.mean(region_data)
            continue
        forecasted_region_data = model.get_forecast(steps=horizon).predicted_mean
        forecasted_values[:, region] = forecasted_region_data
        forecasted_values[:, region] = np.maximum(forecasted_values[:, region], 0)
    return forecasted_values