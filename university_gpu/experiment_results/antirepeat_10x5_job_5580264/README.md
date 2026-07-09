# Local TTT denoising anti-repeat experiment: job 5580264

This folder contains the completed anti-repeat local TTT denoising experiment.

## Purpose

This experiment was run after observing that the local TTT loop repeatedly rediscovered the same plateau solution with MSE 0.21754966515606708.

The goal was to test a smaller and more stable anti-repeat setting using novelty-aware training reward and plateau-avoidance prompt history.

## Method

- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Evaluation mode: official denoising evaluator
- TTT: Yes
- PUCT state reuse: Yes
- LoRA reward update: Yes
- Backend: Hugging Face generation + local LoRA training
- Job ID: 5580264
- Steps: 10
- Rollouts per step: 5
- Total candidates: 50
- Generation temperature: 1.0
- LoRA training max length: 1024
- Slurm memory request: 128G
- Runtime: 00:53:47

## Anti-repeat mechanisms

This run used the following mechanisms:

- exact code duplicate tracking
- metric-level duplicate behavior tracking
- duplicate valid behavior penalty
- novelty-aware training reward
- plateau-avoidance prompt history

## Result summary

- Total candidates: 50
- Valid candidates: 1
- Non-duplicate valid candidates: 1
- Duplicate valid behavior candidates: 0
- Used as duplicate penalty: 0

Best candidate:

- Best source: step_8_rollout_5
- Best MSE: 0.21754966515606708
- Best Poisson: 0.03688785612012908
- Best MSE normalized: 0.28606933832565834
- Best Poisson normalized: 0.9772009063208299
- Best official reward: 4.596651524527119
- Best training reward: 1.459665152452712

## Interpretation

The run completed successfully without OOM.

However, it still rediscovered the same known plateau solution with MSE 0.21754966515606708. Only one valid candidate was found among 50 attempts, so the anti-repeat penalty did not activate in this run.

This suggests that the main limitation in this setting is not only repeated valid behavior, but also the low valid generation rate. The Qwen 1.5B local TTT setup tends to find one strong denoising template, but struggles to discover alternative valid behaviors under this budget.

## Files

- `summary.json`: final summary of the best valid candidate
- `history.json`: full record of all 50 candidate attempts
- `best_magic_denoise.py`: best valid generated candidate
- `ttt_denoise_antirepeat_10x5_5580264.out`: Slurm stdout log
- `ttt_denoise_antirepeat_10x5_5580264.err`: Slurm stderr log
