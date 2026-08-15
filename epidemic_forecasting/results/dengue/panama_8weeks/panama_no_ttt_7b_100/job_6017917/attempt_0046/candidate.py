import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def dengue_forecast(train_values, horizon, **kwargs):
    n_regions = train_values.shape[1]
    forecast = np.zeros((horizon, n_regions))
    for region in range(n_regions):
        region_data = train_values[:, region].reshape(-1, 1)
        if np.all(region_data == region_data[0]):
            forecast[:, region] = region_data[0]
            continue
        model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        model.fit(np.arange(len(region_data)).reshape(-1, 1), region_data)
        future_steps = np.arange(len(region_data), len(region_data) + horizon).reshape(-1, 1)
        forecast[:, region] = np.clip(model.predict(future_steps), 0, None)
    return forecast