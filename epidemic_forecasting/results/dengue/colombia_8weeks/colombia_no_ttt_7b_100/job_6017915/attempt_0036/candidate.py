import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def dengue_forecast(train_values, horizon, random_state=None):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        X = np.arange(train_values.shape[0]).reshape(-1, 1)
        y = train_values[:, region]
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X, y)
        future_X = np.arange(len(y), len(y) + horizon).reshape(-1, 1)
        future_y = model.predict(future_X)
        forecast[:, region] = np.maximum(future_y, 0)
    return forecast