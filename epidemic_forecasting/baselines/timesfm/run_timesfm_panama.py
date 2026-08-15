from pathlib import Path
import pandas as pd

from darts.models import TimesFM2p5Model

from Code.data_loader import load_series
from Code.evaluator import evaluate
from Code.utils import set_seed


# ------------------------------------------------------------
# Configuration matching EpiCastBench
# ------------------------------------------------------------

DATASET = "dengue_panama.csv"

FORECAST_HORIZON = 8

INPUT_CHUNK = 24
OUTPUT_CHUNK = 12
N_EPOCHS = 50
RANDOM_STATE = 42


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

set_seed(RANDOM_STATE)


# ------------------------------------------------------------
# Load the official EpiCastBench-format data
# ------------------------------------------------------------

print("Loading:", DATASET)

series = load_series(DATASET)

print("Series length:", len(series))
print("Number of components:", series.n_components)

train = series[:-FORECAST_HORIZON]
test = series[-FORECAST_HORIZON:]

print("Training observations:", len(train))
print("Test observations:", len(test))


# ------------------------------------------------------------
# TimesFM
# ------------------------------------------------------------

print()
print("Creating TimesFM2p5Model...")

model = TimesFM2p5Model(
    input_chunk_length=INPUT_CHUNK,
    output_chunk_length=OUTPUT_CHUNK,
    n_epochs=N_EPOCHS,
    random_state=RANDOM_STATE,
)

print()
print("Fitting TimesFM...")

model.fit(
    train,
    verbose=True,
)

print()
print("Forecasting 8 weeks...")

prediction = model.predict(
    n=FORECAST_HORIZON,
)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

metrics = evaluate(
    prediction,
    test,
    train,
)

print()
print("=" * 60)
print("EpiCastBench TimesFM — COVID-19 UK — 14 days")
print("=" * 60)

for name, value in metrics.items():
    print(f"{name}: {value}")


# ------------------------------------------------------------
# Save predictions for inspection
# ------------------------------------------------------------

output_dir = Path("timesfm_results")
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

pred_df = prediction.to_dataframe()
actual_df = test.to_dataframe()

pred_df.to_csv(
    output_dir / "dengue_panama_timesfm_predictions.csv"
)

actual_df.to_csv(
    output_dir / "dengue_panama_actual.csv"
)

metrics_df = pd.DataFrame(
    [
        {
            "Dataset": "dengue_panama",
            "Model": "TimesFM",
            "Horizon": FORECAST_HORIZON,
            **metrics,
        }
    ]
)

metrics_df.to_csv(
    output_dir / "dengue_panama_timesfm_metrics.csv",
    index=False,
)

print()
print("Results saved to:")
print(output_dir.resolve())
