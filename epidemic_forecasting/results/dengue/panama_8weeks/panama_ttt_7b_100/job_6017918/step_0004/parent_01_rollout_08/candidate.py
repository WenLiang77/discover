import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from math import inf
from typing import Optional

def dengue_forecast(train_values: np.ndarray, horizon: int, random_state: Optional[int]=None) -> np.ndarray:
    n_regions = train_values.shape[1]
    forecast = np.empty((horizon, n_regions))
    for i in range(n_regions):
        region_data = train_values[:, i]
        if np.any(region_data > 0):
            try:
                model_sarimax = SARIMAX(region_data, order=(1, 1, 0), seasonal_order=(1, 1, 0, 4))
                results_sarimax = model_sarimax.fit(disp=False)
                forecast[i, :] = results_sarimax.forecast(steps=horizon)
            except Exception as e:
                try:
                    model_es = ExponentialSmoothing(region_data, trend='add', seasonal='add', seasonal_periods=4)
                    results_es = model_es.fit()
                    forecast[i, :] = results_es.forecast(steps=horizon)
                except Exception as e:
                    try:
                        model_arima = ARIMA(region_data, order=(1, 1, 0))
                        results_arima = model_arima.fit(disp=False)
                        forecast[i, :] = results_arima.forecast(steps=horizon)
                    except Exception as e:
                        if len(region_data) > 2:
                            X = np.arange(len(region_data)).reshape(-1, 1)
                            y = region_data
                            poly_features = PolynomialFeatures(degree=2)
                            X_poly = poly_features.fit_transform(X)
                            model_lr = LinearRegression()
                            model_lr.fit(X_poly, y)
                            X_pred = np.array(range(len(region_data), len(region_data) + horizon)).reshape(-1, 1)
                            X_pred_poly = poly_features.transform(X_pred)
                            forecast[i, :] = model_lr.predict(X_pred_poly)
                        else:
                            forecast[i, :] = region_data[-1]
        else:
            forecast[i, :] = region_data[-1]
    forecast = np.maximum(forecast, 0)
    return forecast