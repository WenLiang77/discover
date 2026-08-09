# Epidemic Forecasting Results

This directory contains model-generated epidemic forecasting experiments comparing direct language-model generation (**No-TTT**) with **test-time training (TTT)**.

The current COVID-19 experiments use Qwen2.5-Coder-7B-Instruct with a 14-day forecasting horizon. Each condition has a budget of 50 generated candidates.

SMAPE is the primary comparison metric, and **lower is better**.

## COVID-19

### Overall results

| Dataset | Method | Valid | Distinct predictions | Best SMAPE | Job |
|---|---|---:|---:|---:|---:|
| UK | No-TTT | 34/50 | 12 | **39.4883** | 5931327 |
| UK | TTT | 38/50 | 13 | 44.4088 | 5949826 |
| US | No-TTT | 32/50 | 13 | **54.9081** | 5954199 |
| US | TTT | 35/50 | 12 | 55.7518 | 5954200 |

### UK — 14 days

#### Top 10 distinct generated results

To avoid repeated forecasts dominating the ranking, duplicate prediction behaviours are removed from this table.

| Rank | No-TTT candidate | No-TTT SMAPE | TTT candidate | TTT SMAPE |
|---:|---|---:|---|---:|
| 1 | Attempt 26 | 39.4883 | Step 8, Rollout 5 | 44.4088 |
| 2 | Attempt 41 | 45.1765 | Step 10, Rollout 2 | 45.9039 |
| 3 | Attempt 23 | 87.6993 | Step 3, Rollout 3 | 45.9732 |
| 4 | Attempt 39 | 91.1483 | Step 7, Rollout 4 | 46.5118 |
| 5 | Attempt 10 | 91.8048 | Step 6, Rollout 5 | 69.9959 |
| 6 | Attempt 13 | 96.1638 | Step 2, Rollout 1 | 91.8048 |
| 7 | Attempt 3 | 97.7926 | Step 9, Rollout 1 | 96.5771 |
| 8 | Attempt 8 | 98.4814 | Step 1, Rollout 2 | 97.7926 |
| 9 | Attempt 36 | 116.3458 | Step 10, Rollout 5 | 101.4203 |
| 10 | Attempt 1 | 139.2857 | Step 9, Rollout 2 | 102.2854 |

#### TTT stepwise improvement

The table below shows both the best candidate generated within each TTT step and the cumulative best result found up to that point.

| TTT step | Valid candidates | Best in step SMAPE | Best-so-far SMAPE |
|---:|---:|---:|---:|
| 1 | 1 | 97.7926 | **97.7926** |
| 2 | 4 | 91.8048 | **91.8048** |
| 3 | 4 | 45.9732 | **45.9732** |
| 4 | 3 | 45.9732 | **45.9732** |
| 5 | 4 | 45.9732 | **45.9732** |
| 6 | 4 | 45.9732 | **45.9732** |
| 7 | 4 | 45.9732 | **45.9732** |
| 8 | 4 | 44.4088 | **44.4088** |
| 9 | 5 | 44.4088 | **44.4088** |
| 10 | 5 | 45.9039 | **44.4088** |

#### Result directories

- `covid19/uk_14days/no_ttt/job_5931327/`
- `covid19/uk_14days/ttt/job_5949826/`

### US — 14 days

#### Top 10 distinct generated results

To avoid repeated forecasts dominating the ranking, duplicate prediction behaviours are removed from this table.

| Rank | No-TTT candidate | No-TTT SMAPE | TTT candidate | TTT SMAPE |
|---:|---|---:|---|---:|
| 1 | Attempt 48 | 54.9081 | Step 9, Rollout 3 | 55.7518 |
| 2 | Attempt 23 | 55.2188 | Step 6, Rollout 1 | 56.7815 |
| 3 | Attempt 26 | 60.2065 | Step 5, Rollout 2 | 62.4636 |
| 4 | Attempt 9 | 60.4592 | Step 2, Rollout 1 | 62.9713 |
| 5 | Attempt 10 | 62.9713 | Step 5, Rollout 4 | 63.2888 |
| 6 | Attempt 39 | 64.6713 | Step 2, Rollout 4 | 68.2335 |
| 7 | Attempt 46 | 77.8303 | Step 4, Rollout 5 | 117.8252 |
| 8 | Attempt 41 | 80.7350 | Step 9, Rollout 2 | 166.2909 |
| 9 | Attempt 36 | 133.0759 | Step 1, Rollout 2 | 166.2911 |
| 10 | Attempt 13 | 156.5520 | Step 10, Rollout 3 | 166.2911 |

#### TTT stepwise improvement

The table below shows both the best candidate generated within each TTT step and the cumulative best result found up to that point.

| TTT step | Valid candidates | Best in step SMAPE | Best-so-far SMAPE |
|---:|---:|---:|---:|
| 1 | 1 | 166.2911 | **166.2911** |
| 2 | 4 | 62.9713 | **62.9713** |
| 3 | 2 | 62.9713 | **62.9713** |
| 4 | 3 | 62.9713 | **62.9713** |
| 5 | 3 | 62.4636 | **62.4636** |
| 6 | 5 | 56.7815 | **56.7815** |
| 7 | 4 | 56.7815 | **56.7815** |
| 8 | 5 | 56.7815 | **56.7815** |
| 9 | 5 | 55.7518 | **55.7518** |
| 10 | 3 | 55.7518 | **55.7518** |

#### Result directories

- `covid19/us_14days/no_ttt/job_5954199/`
- `covid19/us_14days/ttt/job_5954200/`

### COVID-19 current findings

The current COVID-19 experiments do **not yet show a final best-case accuracy advantage for TTT**. For the UK task, the best No-TTT SMAPE is 39.4883, compared with 44.4088 for TTT. For the US task, the corresponding values are 54.9081 and 55.7518.

However, TTT produced more valid generated programs in both experiments: 38/50 versus 34/50 for the UK, and 35/50 versus 32/50 for the US.

The stepwise tables also show that the TTT search continues to discover stronger candidates during later optimisation steps. This provides evidence that the iterative rollout–evaluation–update loop is active, even though the current best TTT candidates have not yet surpassed the strongest direct-generation samples.

These results therefore support a cautious interpretation: **The current experiments show that TTT performs active within-run optimisation and produced more valid candidates in both COVID-19 runs. However, further experiments are required to determine whether these effects are consistent and whether they translate into a reliable forecasting accuracy advantage over direct generation.**

## Dengue

Dengue forecasting experiments will be added as a second epidemic benchmark and organised separately under `dengue/`.

## Logs

Slurm stdout and stderr logs for the COVID-19 experiments are stored under `logs/covid19/`.

Large intermediate LoRA adapter checkpoints are intentionally excluded from GitHub.

