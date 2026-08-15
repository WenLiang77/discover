import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast Dengue incidence using an ensemble approach combining ARIMA and Bayesian Ridge regression.
    
    Parameters:
        train_values (np.ndarray): Historical dengue incidence data with shape (T, N).
        horizon (int): Number of future time steps to predict.
        kwargs: Additional keyword arguments (not used).
        
    Returns:
        np.ndarray: Forecasted dengue incidence data with shape (horizon, N).
    """
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        train_data = train_values[:-horizon, region]
        val_data = train_values[-horizon:, region]
        scaler = StandardScaler()
        scaler.fit(train_data.reshape(-1, 1))
        scaled_train_data = scaler.transform(train_data.reshape(-1, 1))
        scaled_val_data = scaler.transform(val_data.reshape(-1, 1))
        arima_model = ARIMA(scaled_train_data.flatten(), order=(1, 1, 1))
        arima_fit = arima_model.fit()
        ridge_model = BayesianRidge(alpha=1.0)
        combined_predictions = np.concatenate([arima_fit.predict(steps=horizon), scaled_val_data.flatten()])
        ridge_model.fit(np.arange(len(combined_predictions)).reshape(-1, 1), combined_predictions)
        forecast = ridge_model.predict(np.arange(len(combined_predictions), len(combined_predictions) + horizon).reshape(-1, 1))
        inverse_transformed_forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
        forecasted_values[:, region] = inverse_transformed_forecast.clip(min=0)
    return forecasted_values