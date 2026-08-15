import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def dengue_forecast(train_values, horizon, **kwargs):
    """
    Forecast dengue incidence using linear regression with feature engineering.
    
    Args:
        train_values (np.ndarray): Training data of shape (T, N).
        horizon (int): Number of future time steps to predict.
        
    Returns:
        np.ndarray: Forecasted values of shape (horizon, N).
    """
    T, N = train_values.shape
    forecast = np.zeros((horizon, N))
    for n in range(N):
        X = np.arange(T).reshape(-1, 1)
        y = train_values[:, n]
        lags = [1, 2, 4, 8]
        features = []
        for lag in lags:
            features.append(np.roll(y, -lag, axis=0))
        features.append(y)
        features = np.array(features)
        features[np.isnan(features)] = 0
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        features_scaled = scaler.transform(features.T).T
        combined_features = np.hstack([X_scaled, features_scaled])
        model = LinearRegression()
        try:
            model.fit(combined_features[:-lag], y[lag:])
        except Exception as e:
            print(f'Error fitting model for region {n}: {e}')
            continue
        future_X = np.array(range(T, T + horizon)).reshape(-1, 1)
        future_combined_features = np.hstack([scaler.transform(future_X), scaler.transform(features[-lag:].T)])
        forecast[:horizon, n] = model.predict(future_combined_features)
    forecast[forecast < 0] = 0
    return forecast