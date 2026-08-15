import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast Dengue incidence for multiple regions over a given horizon.
    
    Parameters:
        train_values (numpy.ndarray): A (T, N) array of historical dengue cases.
        horizon (int): Number of future time steps to predict.
        kwargs: Additional keyword arguments (not used).
        
    Returns:
        numpy.ndarray: A (horizon, N) array of predicted dengue cases.
    """
    train_values = np.array(train_values)
    forecast = np.zeros((horizon, train_values.shape[1]))
    for i in range(train_values.shape[1]):
        region_data = train_values[:, i]
        if len(region_data) < 2:
            continue
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(region_data.reshape(-1, 1))
        try:
            arima_model = ARIMA(scaled_data.flatten(), order=(5, 1, 0), seasonal_order=(1, 1, 0, 7))
            arima_results = arima_model.fit(disp=False)
        except Exception:
            arima_results = None
        if arima_results is None:
            try:
                es_model = ExponentialSmoothing(scaled_data.flatten(), trend='add', seasonal='add', seasonal_periods=7).fit(disp=False)
            except Exception:
                continue
            else:
                forecast[:horizon, i] = es_model.forecast(horizon)
                forecast[:horizon, i] = np.clip(forecast[:horizon, i], 0, None)
                forecast[:horizon, i] = scaler.inverse_transform(forecast[:horizon, i].reshape(-1, 1))[:, 0]
                continue
        predictions = arima_results.forecast(steps=horizon)
        predictions = np.clip(predictions, 0, None)
        forecast[:horizon, i] = scaler.inverse_transform(predictions.reshape(-1, 1))[:, 0]
    return forecast