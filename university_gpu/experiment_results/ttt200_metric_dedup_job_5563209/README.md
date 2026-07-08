# Local TTT denoising experiment with metric-level duplicate filtering: job 5563209

This folder contains the local TTT-Discover-style 200-candidate denoising result with metric-level duplicate behavior filtering.

## Purpose

This experiment was run after observing that many generated candidates produced the same best MSE even when their code strings were not exactly identical.

The goal was to detect repeated valid behavior using the metric signature, not only exact code hashes.

## Method

- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Evaluation mode: official denoising evaluator
- Attempts: 200
- TTT steps: 20
- Rollouts per step: 10
- TTT: Yes
- PUCT state reuse: Yes
- LoRA reward update: Yes
- Backend: Hugging Face generation + local LoRA training
- Job ID: 5563209
- Slurm memory request: 128G
- Runtime: 02:29:26
- Generation temperature: 0.9
- LoRA training max length: 2048
- Duplicate filtering:
  - exact valid code duplicate detection
  - metric-level valid behavior duplicate detection
  - duplicate valid behavior was skipped from PUCT child creation and LoRA training

Note: this run was completed before the later anti-repeat penalty and plateau-avoidance prompt changes.

## Result summary

- Total candidates: 200
- Valid candidates: 10
- Duplicate valid code candidates: 5
- Duplicate valid metric candidates: 9
- Duplicate valid behavior candidates: 9
- Unique valid behavior candidates: 1

Best candidate:

- Best source: step_1_rollout_1
- Best MSE: 0.21754966515606708
- Best Poisson: 0.03688785612012908
- Best MSE normalized: 0.28606933832565834
- Best Poisson normalized: 0.9772009063208299
- Best reward: 4.596651524527119

## Interpretation

The metric-level duplicate filter worked as intended: 9 out of 10 valid candidates were identified as duplicate behavior.

However, the run still did not discover a new unique valid solution beyond the previously found best candidate with MSE 0.21754966515606708.

This suggests that the local TTT loop remains strongly attracted to the known high-reward denoising template. Metric-level filtering can detect and skip repeated behavior, but skipping duplicates alone is not enough to push the model toward a better solution.

## Files

- `summary.json`: final summary of the best valid candidate
- `history.json`: full record of the 200 candidate attempts
- `best_magic_denoise.py`: best valid generated candidate
- `ttt_denoise_200_5563209.out`: Slurm stdout log
- `ttt_denoise_200_5563209.err`: Slurm stderr log
