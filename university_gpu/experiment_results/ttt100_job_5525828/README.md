# Local TTT denoising experiment: job 5525828

This folder contains the local TTT-Discover-style 100-candidate result for the official denoising experiment after prompt tightening.

## Purpose

This experiment was run to test whether the local TTT-Discover-style loop can improve over the Qwen-only baseline when using the same Qwen model and the same official Section 4.4 denoising evaluator.

The run uses local Hugging Face generation, PUCT-style state reuse, and LoRA reward-based updates instead of the original Tinker backend.

## Method

- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Evaluation mode: official denoising evaluator
- Attempts: 100
- TTT: Yes
- PUCT state reuse: Yes
- LoRA reward update: Yes
- Backend: Hugging Face generation + local LoRA training
- Job ID: 5525828
- Best source: step_4_rollout_1

Each rollout generates a candidate `magic_denoise` implementation. The candidate is evaluated using the official denoising evaluator, and valid candidates are judged using the official validity check. The TTT loop then uses reward signals to update the local LoRA adapter.

## Result summary

- Total candidates: 100
- Valid candidates: 1
- Valid rate: 0.01
- Best source: step_4_rollout_1
- Best MSE: 0.31875574507308685
- Best Poisson: 0.03697139036266695
- Best Poisson normalized: 0.9768310173636325
- Best reward: 3.1371983578545763

## Comparison with previous runs

Previous Qwen-only baseline from job 5522540:

- Total candidates: 50
- Valid candidates: 1
- Best MSE: 0.4021363015190825
- Best Poisson normalized: 0.9768564760043417
- Best reward: 2.48671904581225

Previous local TTT 50-candidate result from job 5443238:

- Total candidates: 50
- Valid candidates: 1
- Best MSE: 0.21754966515606708
- Best Poisson normalized: 0.9772009063208299
- Best reward: 4.596651524527119

In this single-run comparison, the 100-candidate local TTT run improved over the Qwen-only baseline in Best MSE and reward, but it did not outperform the earlier 50-candidate TTT run. This suggests that the local TTT-style setup can find better candidates than direct Qwen-only generation, but the result is still unstable and sensitive to randomness.

More seeds and repeated runs are needed before making a stronger performance claim.

## Files

- `summary.json`: final summary of the best valid candidate
- `history.json`: full record of the 100 candidate attempts
- `best_magic_denoise.py`: best valid generated candidate
- `ttt_denoise_100_5525828.out`: Slurm stdout log
- `ttt_denoise_100_5525828.err`: Slurm stderr log
