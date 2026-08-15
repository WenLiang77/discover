import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecast = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        poly_features = PolynomialFeatures(degree=3)
        model = make_pipeline(poly_features, LinearRegression())
        model.fit(np.arange(len(region_data)).reshape(-1, 1), region_data)
        forecast[:horizon, region] = model.predict(np.arange(len(region_data), len(region_data) + horizon).reshape(-1, 1))
    forecast = np.clip(forecast, a_min=0, a_max=None)
    return forecast