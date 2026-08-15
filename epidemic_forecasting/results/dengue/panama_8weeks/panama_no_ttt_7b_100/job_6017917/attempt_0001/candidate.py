import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import smape, mae, rmse, mase
import matplotlib.pyplot as plt

def dengue_forecast(train_values, horizon, **kwargs):

    def forecast_region(region_data, horizon):
        try:
            model = SARIMAX(region_data, order=(1, 0, 1), seasonal_order=(1, 0, 1, 52))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            return forecast
        except Exception as e:
            print(f'Error fitting model: {e}')
            return np.zeros(horizon)
    forecast_array = np.zeros((horizon, train_values.shape[1]))
    for region in range(train_values.shape[1]):
        region_data = train_values[:, region]
        region_forecast = forecast_region(region_data, horizon)
        forecast_array[:, region] = region_forecast
    forecast_array = np.maximum(forecast_array, 0)
    return forecast_array