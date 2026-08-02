# Epidemic Forecasting Experiment Plan

## Place the files

Copy the bundle contents into the existing repository:

```text
epidemic_forecasting/
├── run_ttt.py
├── summarize_results.py
└── slurm/
    ├── 01_qwen_uk_smoke.slurm
    ├── 02_ttt_uk_smoke.slurm
    ├── 03_qwen_uk_50.slurm
    ├── 04_ttt_uk_50.slurm
    ├── 05_qwen_us_50.slurm
    └── 06_ttt_us_50.slurm
```

The scripts use `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
Do not switch to 7B until both smoke tests have completed without
out-of-memory errors.

## Local checks

From the repository root:

```bash
python -m py_compile epidemic_forecasting/run_ttt.py
python -m py_compile epidemic_forecasting/summarize_results.py

python -m epidemic_forecasting.run_ttt \
  --dataset uk \
  --forecast-horizon 14 \
  --dry-run
```

The dry run does not load Qwen. It checks the task, evaluator,
output directory, initial state and PUCT persistence.

## Server order

Run the scripts in this order. Do not submit all jobs together.

```bash
sbatch epidemic_forecasting/slurm/01_qwen_uk_smoke.slurm
sbatch epidemic_forecasting/slurm/02_ttt_uk_smoke.slurm
```

Inspect both `summary.json` files before continuing.

Then run the matched 50-candidate UK comparison:

```bash
sbatch epidemic_forecasting/slurm/03_qwen_uk_50.slurm
sbatch epidemic_forecasting/slurm/04_ttt_uk_50.slurm
```

Only after the UK runs are stable, run the US comparison:

```bash
sbatch epidemic_forecasting/slurm/05_qwen_us_50.slurm
sbatch epidemic_forecasting/slurm/06_ttt_us_50.slurm
```

## Fair comparison

The formal Qwen-only and TTT scripts both evaluate 50 generated
candidates:

- Qwen-only: 50 independent generations.
- TTT: 10 steps × 1 selected parent × 5 rollouts = 50 candidates.

Both use the same model, dataset, 14-day horizon, seed,
temperature, top-p, token limits, evaluator and reward.

The primary comparison keeps `--duplicate-penalty 0`. A novelty
penalty should be a separate ablation experiment rather than mixed
into the main TTT result.

## Compare completed runs

```bash
python -m epidemic_forecasting.summarize_results
```

This creates:

```text
epidemic_forecasting/results/experiment_comparison.csv
```

## Important limitation

This runner uses the repository's local LoRA loss and PUCT modules.
It is a local approximation to TTT-Discover rather than a
bit-for-bit reproduction of the original Tinker/RL backend.
