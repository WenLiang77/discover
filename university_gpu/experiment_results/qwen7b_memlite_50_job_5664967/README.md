# Qwen7B memlite TTT denoising experiment: job 5664967

## Configuration

- Model: Qwen/Qwen2.5-Coder-7B-Instruct
- TTT setting: memory-lite local TTT
- Candidates: 50
- Steps: 10
- Rollouts per step: 5
- LoRA rank: 4
- Training max length: 1024
- Job ID: 5664967

## Result

- Status: Completed
- Best MSE: 0.21754966515606708
- Best reward: 4.596651524527119
- Best source: step_7_rollout_5
- Valid candidates: 9 / 50
- Valid MSE distribution:
  - 0.217549665156: 8 candidates
  - 0.321921832596: 1 candidate

## Interpretation

The Qwen7B memory-lite local TTT run successfully completed and improved over the Qwen7B-only baseline. However, it again converged to the known plateau solution with MSE 0.21755. Most valid candidates repeated the same behavior.
