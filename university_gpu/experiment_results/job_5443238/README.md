# Local TTT-Discover Official Denoising Experiment

## Experiment setting

- Job ID: 5443238
- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Evaluation mode: official denoising evaluator
- Steps: 5
- Rollouts per step: 10
- Total candidates: 50
- Hardware: Isambard-AI GH200 GPU

## Best result

The run successfully found a valid denoising function under the official evaluator.

From `summary.json`:

- Valid: true
- MSE: 0.21754966515606708
- Poisson: 0.03688785612012908
- Reward: 4.596651524527119
- MSE normalized: 0.28606933832565834
- Poisson normalized: 0.9772009063208299
- Source: step_2_rollout_1

The best generated function is saved in:

```text
best_magic_denoise.py

```

## Files

- `summary.json`: final best result summary.
- `best_magic_denoise.py`: best generated denoising function.
- `history.json`: full rollout history, including generated candidates and evaluation results.

## Step-wise result

The experiment completed successfully. It evaluated 50 candidates in total. The best valid candidate was found at `step_2_rollout_1`.

The best valid MSE remained:

```text
0.21754966515606708
```

## Note

This is a local university-GPU version inspired by TTT-Discover. It uses HuggingFace + LoRA, local PUCT reuse, and the official denoising evaluator. It is not yet a full reproduction of the original TTT-Discover Tinker-based importance-sampling training pipeline.
