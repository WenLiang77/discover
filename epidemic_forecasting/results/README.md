# Epidemic Forecasting Results

This directory contains model-generated epidemic forecasting experiments.

Results are organised first by disease, then by dataset / forecasting horizon,
and finally by whether test-time training (TTT) is used.

## COVID-19

The current COVID-19 experiments use the 14-day medium-term forecasting setting.

### UK — 14 days

| Method | Model | Generations | Valid | Best SMAPE | Job |
|---|---|---:|---:|---:|---|
| No-TTT | Qwen2.5-Coder-7B-Instruct | 50 | 34/50 | 39.4883 | 5931327 |
| TTT | Qwen2.5-Coder-7B-Instruct | 50 | 38/50 | 44.4088 | 5949826 |

Result directories:

- `covid19/uk_14days/no_ttt/job_5931327/`
- `covid19/uk_14days/ttt/job_5949826/`

### US — 14 days

| Method | Model | Generations | Valid | Best SMAPE | Job |
|---|---|---:|---:|---:|---|
| No-TTT | Qwen2.5-Coder-7B-Instruct | 50 | 32/50 | 54.9081 | 5954199 |
| TTT | Qwen2.5-Coder-7B-Instruct | 50 | 35/50 | 55.7518 | 5954200 |

Result directories:

- `covid19/us_14days/no_ttt/job_5954199/`
- `covid19/us_14days/ttt/job_5954200/`

## Dengue

Dengue forecasting experiments will be added as a separate benchmark.

## Logs

Slurm stdout and stderr logs are stored under:

- `logs/covid19/`
- `logs/dengue/` for future dengue experiments

## Notes

- Lower SMAPE is better.
- No-TTT results contain independently generated forecasting programs.
- TTT results contain iterative model-generated candidates with local LoRA
  updates and PUCT-based search.
- Large intermediate LoRA adapter checkpoints are not stored in GitHub.
