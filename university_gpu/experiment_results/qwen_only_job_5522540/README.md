# Qwen-only denoising baseline: job 5522540

This folder contains the Qwen-only baseline result for the official denoising experiment.

## Purpose

This experiment was run as a baseline comparison against the local TTT-Discover-style experiment.

The goal was to test what happens if we use the same Qwen model and the same official denoising evaluator, but do not use the TTT-style components.

## Method

- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Evaluation mode: official denoising evaluator
- Attempts: 50
- TTT: No
- PUCT state reuse: No
- LoRA reward update: No
- Backend: HuggingFace generation only
- Job ID: 5522540

Each attempt starts from the same initial `magic_denoise` implementation and generates one candidate using the base Qwen model. The generated candidate is then evaluated using the official denoising evaluator.

## Result summary

- Total candidates: 50
- Valid candidates: 1
- Valid rate: 0.02
- Best source: attempt_17
- Best MSE: 0.4021363015190825
- Best Poisson: 0.0369656408850835
- Best Poisson normalized: 0.9768564760043417
- Best reward: 2.48671904581225

## Comparison with local TTT 50-candidate run

Previous local TTT-style result from job 5443238:

- Total candidates: 50
- Best MSE: 0.21754966515606708
- Best Poisson normalized: 0.9772009063208299
- Best reward: 4.596651524527119

In this single-run comparison, the local TTT-style experiment found a substantially better best candidate than the Qwen-only baseline under the same 50-candidate budget.

However, this is still only one run. Multiple seeds and more ablation experiments are needed before making a stronger performance claim.

## Files

- `summary.json`: final summary of the Qwen-only baseline
- `history.json`: full record of all 50 attempts
- `best_magic_denoise.py`: best valid generated candidate
- `qwen_only_denoise_5522540.out`: Slurm stdout log
- `qwen_only_denoise_5522540.err`: Slurm stderr log
