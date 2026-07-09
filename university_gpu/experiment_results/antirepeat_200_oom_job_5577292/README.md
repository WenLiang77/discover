# Local TTT denoising anti-repeat partial experiment: job 5577292

This folder contains a failed but informative anti-repeat local TTT denoising experiment.

## Purpose

This experiment tested the anti-repeat version with the larger 20-step and 10-rollout setting, giving a planned budget of 200 candidates.

The run was designed to test whether duplicate valid behavior could be used as a low-reward anti-repeat signal during LoRA training.

## Method

- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Evaluation mode: official denoising evaluator
- Planned attempts: 200
- Planned steps: 20
- Planned rollouts per step: 10
- TTT: Yes
- PUCT state reuse: Yes
- LoRA reward update: Yes
- Backend: Hugging Face generation + local LoRA training
- Job ID: 5577292
- Slurm memory request: 128G

## Result summary before failure

The job was OOM-killed before completion.

Partial history:

- Completed candidates before failure: 60
- Last completed rollout: step 6 rollout 10
- Valid candidates: 3
- Duplicate valid behavior candidates: 2
- Used as duplicate penalty: 2

Best candidate before failure:

- Best source: step_4_rollout_6
- Best MSE: 0.21754966515606708
- Best Poisson normalized: 0.9772009063208299
- Best official reward: 4.596651524527119
- Best training reward: 1.459665152452712

## Failure

The job failed with an OOM kill event:

- Slurm state: OUT_OF_MEMORY
- Exit code: 0:125

## Interpretation

The anti-repeat penalty mechanism activated successfully: duplicate valid behavior was detected and used as a low-reward penalty signal.

However, the larger 20x10 setting was not stable under the available memory budget after duplicate examples were included in LoRA training. The run still rediscovered the known plateau solution with MSE 0.21754966515606708 before failing.

This motivated the smaller 10x5 anti-repeat experiment in job 5580264.

## Files

- `history.json`: partial record before OOM
- `ttt_denoise_200_5577292.out`: Slurm stdout log
- `ttt_denoise_200_5577292.err`: Slurm stderr log
