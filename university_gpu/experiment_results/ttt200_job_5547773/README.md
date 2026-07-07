# Local TTT denoising experiment: job 5547773

This folder contains the local TTT-Discover-style 200-candidate result for the official denoising experiment.

## Purpose

This experiment was run to test whether increasing the local TTT search budget improves the denoising result.

The configuration uses 20 TTT steps and 10 rollouts per step, giving 200 generated candidates in total.

## Method

- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Evaluation mode: official denoising evaluator
- Attempts: 200
- TTT: Yes
- PUCT state reuse: Yes
- LoRA reward update: Yes
- Backend: Hugging Face generation + local LoRA training
- Job ID: 5547773
- Slurm memory request: 128G
- Runtime: 01:28:11
- Best source: step_7_rollout_6

Each rollout generates a candidate `magic_denoise` implementation. The candidate is evaluated using the official denoising evaluator, and valid candidates are judged using the official validity check.

## Result summary

- Best MSE: 0.21754966515606708
- Best Poisson: 0.03688785612012908
- Best MSE normalized: 0.28606933832565834
- Best Poisson normalized: 0.9772009063208299
- Best reward: 4.596651524527119
- Best source: step_7_rollout_6

## Comparison with previous runs

Qwen-only baseline from job 5522540:

- Attempts: 50
- Best MSE: 0.4021363015190825
- Best Poisson normalized: 0.9768564760043417
- Best reward: 2.48671904581225

Previous local TTT 50-candidate result from job 5443238:

- Attempts: 50
- Best MSE: 0.21754966515606708
- Best Poisson normalized: 0.9772009063208299
- Best reward: 4.596651524527119

Previous local TTT 100-candidate result from job 5525828:

- Attempts: 100
- Best MSE: 0.31875574507308685
- Best Poisson normalized: 0.9768310173636325
- Best reward: 3.1371983578545763

This 200-candidate run successfully rediscovered the best candidate found in the earlier TTT-50 run. It outperforms the Qwen-only baseline and the TTT-100 run, but does not improve beyond the previous best TTT result.

This suggests that the local TTT-style setup can find strong candidates, but the improvement is not monotonic with more attempts and remains sensitive to randomness.

## Files

- `summary.json`: final summary of the best valid candidate
- `history.json`: full record of the 200 candidate attempts
- `best_magic_denoise.py`: best valid generated candidate
- `ttt_denoise_200_5547773.out`: Slurm stdout log
- `ttt_denoise_200_5547773.err`: Slurm stderr log
