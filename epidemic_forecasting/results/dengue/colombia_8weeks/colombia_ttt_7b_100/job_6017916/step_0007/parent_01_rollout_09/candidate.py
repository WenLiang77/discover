import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions), dtype=np.float64)
    for region in range(num_regions):
        region_data = train_values[:, region]
        _, p_value, _, _, _ = adfuller(region_data)
        is_stationary = p_value < 0.05
        if is_stationary:
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 52)
        else:
            decomposition = seasonal_decompose(region_data, model='additive')
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            trend_order = (1, 1)
            seasonal_order = (1, 1, 1, 52)
            model = SARIMAX(trend, order=trend_order, seasonal_order=seasonal_order).fit(disp=False)
            residual_order = model.order + model.seasonal_order
            residual_model = SARIMAX(residual, order=residual_order, seasonal_order=(0, 0, 0, 0)).fit(disp=False)
            fitted_resid = residual_model.predict(start=0, end=len(region_data) - 1)
            forecasted_region_data = trend[-1] * seasonal[-1] * fitted_resid[-1]
            forecasted_values[:, region] = forecasted_region_data