# Local GPU TTT-Discover denoising experiment results

This folder collects local university-GPU experiments for the official denoising benchmark inspired by TTT-Discover Section 4.4.

## Result table

| Experiment | Job ID | Candidates | TTT | Main modification | Status | Valid candidates | Best MSE | Poisson norm | Reward | Notes |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| Qwen-only baseline | 5522540 | 50 | No | Base Qwen generation only | Completed | 1 | 0.402136 | 0.976856 | 2.486719 | Baseline without TTT |
| TTT 50 | 5443238 | 50 | Yes | Local TTT + PUCT + LoRA | Completed | 1 | 0.217550 | 0.977201 | 4.596652 | Best overall result |
| TTT 100 | 5525828 | 100 | Yes | Local TTT + PUCT + LoRA | Completed | 1 | 0.318756 | 0.976831 | 3.137198 | Worse than TTT50 |
| TTT 200 | 5547773 | 200 | Yes | 20 steps x 10 rollouts | Completed | 111 | 0.217550 | 0.977201 | 4.596652 | Rediscovered TTT50 best; many repeated behaviors |
| TTT 200 metric dedup | 5563209 | 200 | Yes | Metric-level duplicate filtering | Completed | 10 | 0.217550 | 0.977201 | 4.596652 | 9/10 valid candidates were duplicate behavior |
| TTT 200 anti-repeat | 5577292 | 200 planned | Yes | Duplicate behavior as low-reward penalty | OOM | 3 before failure | 0.217550 | 0.977201 | 4.596652 | Failed after 60 candidates |
| TTT 10x5 anti-repeat | 5580264 | 50 | Yes | Smaller anti-repeat run | Completed | 1 | 0.217550 | 0.977201 | 4.596652 | Completed but did not find new behavior |

## Main observations

The local TTT setup consistently outperforms the Qwen-only baseline when it finds the strong plateau solution with MSE 0.21755.

However, repeated experiments show that the local setup tends to rediscover the same denoising behavior rather than steadily improving beyond it. Metric-level duplicate filtering confirmed that many valid candidates were behavior duplicates, even when their code strings were not exactly identical.

The anti-repeat penalty version worked mechanically, but did not improve the best score. In the larger 20x10 setting it caused an OOM failure, while the smaller 10x5 setting completed but found only one valid candidate.

Overall, the experiments suggest that the local Qwen 1.5B + LoRA implementation can discover a strong solution, but has limited exploration ability and does not show monotonic improvement with more attempts.
