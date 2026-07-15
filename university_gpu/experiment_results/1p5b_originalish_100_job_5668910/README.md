# 1.5B originalish TTT denoising experiment: job 5668910

## Configuration

- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- TTT setting: originalish local trainer
- Candidates: 100
- Steps: 10
- Rollouts per step: 10
- LoRA rank: 8
- Training max length: 1024
- KL coefficient: 0.01
- Group size: 10
- Job ID: 5668910

## Result

- Status: Completed
- Total candidates: 100
- Valid candidates: 2
- Valid rate: 0.02
- Best source: step_1_rollout_9
- Best MSE: 0.21754966515606708
- Best Poisson normalized: 0.9772009063208299
- Best reward: 4.596651524527119

Valid MSE distribution:

- 0.217549665156: 1 candidate
- 0.304721499343: 1 candidate

## Interpretation

The originalish local trainer reduced repeated plateau behavior compared with earlier local TTT runs, but it did not improve the best score. It found only two valid candidates out of 100 attempts, and the best remained the known plateau solution with MSE 0.21755.

This suggests that the local originalish update does not reproduce the stronger continuous optimization behavior of the original TTT-Discover backend. It reduces repetition but also has a low valid generation rate.
