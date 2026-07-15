# Qwen7B originalish TTT denoising experiment: job 5667201

## Configuration

- Model: Qwen/Qwen2.5-Coder-7B-Instruct
- TTT setting: originalish local trainer
- Candidates planned: 50
- Steps: 10
- Rollouts per step: 5
- LoRA rank: 4
- Training max length: 1024
- KL coefficient: 0.01
- Job ID: 5667201

## Result

- Status: Failed
- Failure reason: OOM during originalish TTT update

## Interpretation

The originalish Qwen7B trainer was more memory-intensive than the memlite version because it computes old log probabilities, reference/KL log probabilities, and an importance-style update. Under the available single-GPU memory budget, this run was OOM-killed.
