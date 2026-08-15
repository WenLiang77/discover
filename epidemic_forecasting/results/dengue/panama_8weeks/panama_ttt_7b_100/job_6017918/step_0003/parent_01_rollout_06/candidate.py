import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast Dengue incidence using a hybrid approach combining SARIMAX, Exponential Smoothing,
    and Linear Regression models for each region.
    
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
        if np.all(np.diff(train_values[:, region]) == 0):
            forecasted_values[:, region] = train_values[-1, region]
        else:
            try:
                sarima_model = SARIMAX(train_values[:, region], order=(2, 1, 0), seasonal_order=(1, 1, 1, 52))
                sarima_fit = sarima_model.fit(disp=False)
                sarima_forecast = sarima_fit.forecast(steps=horizon)
                ets_model = ExponentialSmoothing(train_values[:, region], trend='add', seasonal='mul', seasonal_periods=52)
                ets_fit = ets_model.fit(use_boxcox=True)
                ets_forecast = ets_fit.forecast(steps=horizon)
                combined_features = np.column_stack((sarima_forecast, ets_forecast))
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(combined_features)
                lr_model = LinearRegression()
                lr_model.fit(scaled_features, train_values[:, region])
                combined_forecast = np.column_stack((sarima_forecast, ets_forecast))
                scaled_combined_forecast = scaler.transform(combined_forecast)
                predicted_values = lr_model.predict(scaled_combined_forecast)
                forecasted_values[:, region] = predicted_values.clip(min=0)
            except Exception as e:
                ets_fallback_model = ExponentialSmoothing(train_values[:, region], trend='add', seasonal='mul', seasonal_periods=52)
                ets_fallback_fit = ets_fallback_model.fit(use_boxcox=True)
                forecasted_values[:, region] = ets_fallback_fit.forecast(steps=horizon).clip(min=0)
    return forecasted_values