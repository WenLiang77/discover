import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, symmetric_mean_absolute_percentage_error

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        stationarity_test = adfuller(region_data)
        is_stationary = stationarity_test[1] > 0.05
        if not is_stationary:
            diff_order = 1
            region_data = np.diff(region_data, n=diff_order)
        try:
            model = SARIMAX(region_data, order=(1, diff_order, 1), seasonal_order=(1, 1, 1, 52)).fit(disp=False)
        except Exception as e:
            model = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=52).fit(disp=False)
        forecasted_region_data = model.forecast(steps=horizon)
        if not is_stationary:
            forecasted_region_data = np.cumsum(forecasted_region_data)
            forecasted_region_data = np.insert(forecasted_region_data, 0, region_data[-1])
        forecasted_region_data = np.maximum(0, forecasted_region_data)
        forecasted_values[:, region] = forecasted_region_data
    return forecasted_values