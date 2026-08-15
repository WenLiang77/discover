import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def dengue_forecast(train_values, horizon, **kwargs):
    num_regions = train_values.shape[1]
    forecasted_values = np.zeros((horizon, num_regions))
    for region in range(num_regions):
        region_data = train_values[:, region]
        if len(region_data) < 10 or np.sum(region_data == 0) / len(region_data) > 0.9:
            forecasted_values[:, region] = np.mean(region_data)
            continue
        split_point = int(0.8 * len(region_data))
        train_data, val_data = (region_data[:split_point], region_data[split_point:])
        pipeline = make_pipeline(StandardScaler(), ARIMA(order=(1, 1, 1)), BayesianRidge(alpha_1=0.001, alpha_2=0.001))
        pipeline.fit(train_data.reshape(-1, 1), train_data)
        try:
            val_predictions = pipeline.predict(val_data.reshape(-1, 1))
            future_predictions = pipeline.predict(np.array(range(len(val_data), len(val_data) + horizon)).reshape(-1, 1))
        except Exception as e:
            forecasted_values[:, region] = np.mean(region_data)
            continue
        future_predictions = np.maximum(future_predictions, 0)
        forecasted_values[:, region] = future_predictions
    return forecasted_values