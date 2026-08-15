import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        region_data = train_values[:, region]
        if len(region_data) < 2 or np.std(region_data) == 0:
            moving_avg = np.convolve(region_data, np.ones(4) / 4, mode='valid')
            forecast[:len(moving_avg), region] = moving_avg[-1]
        else:
            try:
                model_es = ExponentialSmoothing(region_data, trend='add', seasonal='multiplicative', seasonal_periods=52)
                fit_model_es = model_es.fit()
                forecast[:fit_model_es.fittedvalues.size, region] = fit_model_es.forecast(horizon)[:fit_model_es.fittedvalues.size]
                if fit_model_es.aic > 1000:
                    model_sarima = SARIMAX(region_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
                    fit_model_sarima = model_sarima.fit(disp=False)
                    forecast[:fit_model_sarima.fittedvalues.size, region] = fit_model_sarima.forecast(horizon)[:fit_model_sarima.fittedvalues.size]
            except Exception as e:
                moving_avg = np.convolve(region_data, np.ones(4) / 4, mode='valid')
                forecast[:len(moving_avg), region] = moving_avg[-1]
    return np.clip(forecast, 0, None)