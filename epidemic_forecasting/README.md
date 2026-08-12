# Epidemic Forecasting with LLM and Test-Time Training

This project investigates the use of large language models (LLMs) and test-time training (TTT) for epidemic forecasting.

The current experiments focus on **14-day COVID-19 forecasting** for the UK and US using **Qwen2.5-Coder-7B-Instruct**.

Two settings are compared:

- **No-TTT:** 100 independent code generations.
- **TTT:** 10 steps × 10 rollouts = 100 generated candidates.

Generated forecasting code is evaluated primarily using **SMAPE**, where lower values indicate better forecasting performance.

---

## COVID-19 Results

| Dataset | Method | Best SMAPE ↓ | Valid Candidates | Distinct Valid |
|---|---|---:|---:|---:|
| UK | No-TTT | **39.48** | 51 / 100 | 22 |
| UK | TTT | 46.95 | **63 / 100** | **29** |
| US | No-TTT | **54.91** | 49 / 100 | **23** |
| US | TTT | 55.84 | **70 / 100** | 14 |

TTT generated more valid candidates on both datasets. However, the best SMAPE was still achieved by No-TTT for both the UK and US experiments.

---

## UK 14-Day Forecasts

The UK dataset contains four regional time series: **England, Northern Ireland, Scotland and Wales**.

Each figure compares the predicted number of cases with the true held-out observations over the 14-day forecasting period.

### No-TTT

The best UK No-TTT candidate achieved a SMAPE of **39.48**.

![UK No-TTT forecast](results/covid19/final_100/figures/uk_no_ttt_forecast.png)

### TTT

The best UK TTT candidate achieved a SMAPE of **46.95**.

![UK TTT forecast](results/covid19/final_100/figures/uk_ttt_forecast.png)

The figures provide a direct view of the daily forecasting behaviour and make it possible to compare the predicted epidemic trajectory with the real observations.

---

## US 14-Day Forecasts

The US dataset contains 56 regional time series. To keep the visualisation readable, four representative regions are shown: **California, Florida, New York and Texas**.

### No-TTT

The best US No-TTT candidate achieved a SMAPE of **54.91**.

![US No-TTT forecast](results/covid19/final_100/figures/us_no_ttt_forecast.png)

### TTT

The best US TTT candidate achieved a SMAPE of **55.84**.

![US TTT forecast](results/covid19/final_100/figures/us_ttt_forecast.png)

The US dataset contains many zero observations and occasional large reporting spikes, making accurate day-by-day forecasting particularly challenging.

---

## Saved COVID-19 Results

The final 100-candidate results are stored in:

```text
results/covid19/final_100/
├── uk_no_ttt/
├── uk_ttt/
├── us_no_ttt/
└── us_ttt/
```

Each experiment folder contains the generated forecasting code and evaluation results.

Important files include:

- `aggregate_results.json` — overall experiment statistics
- `best_generated_candidate.py` — best generated forecasting code
- `best_forecast_vs_actual.csv` — predicted and actual values for each forecast day
- `best_series_metrics.csv` — evaluation results for individual time series
- `top10_by_smape.csv` — top distinct candidates ranked by SMAPE
- `top10_by_reward.csv` — top candidates ranked by optimisation reward

---

## EpiCastBench Baseline Comparison

The next stage of the COVID-19 experiments will compare the LLM-generated forecasting methods with established forecasting baselines from **EpiCastBench**.

Planned baseline methods include:

- TimesFM
- DLinear
- Random Forest
- XGBoost

These methods will be evaluated using the same local COVID-19 data and forecasting pipeline to provide a more direct comparison with the generated models.

---

## Dengue Forecasting

The next epidemic forecasting task will extend the same framework to dengue data.

### Results

Results will be added here after the dengue experiments are completed.

### Forecast Visualisation

Forecast visualisations will be added here after the dengue experiments are completed.
