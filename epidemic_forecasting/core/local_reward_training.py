import math
from typing import List, Optional

import torch


def compute_entropic_adaptive_beta_advantages(
    rewards: List[float],
    delta: float = math.log(2.0),
    beta_max: float = 1e6,
    iters: int = 60,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Local PyTorch version of the original TTT-Discover entropic_adaptive_beta advantage.

    Original idea:
    - Given rewards inside one rollout group.
    - Find beta such that KL(q_beta || uniform) is approximately log(2).
    - q_beta is proportional to exp(beta * reward).
    - Then compute leave-one-out entropic advantage:
        advantage_i = exp(beta r_i) / mean_{j != i} exp(beta r_j) - 1

    Higher reward gets positive advantage.
    Lower reward gets negative advantage.

    If there is only one sample, or all rewards are the same, return zero advantages.
    """

    if len(rewards) == 0:
        raise ValueError("rewards cannot be empty")

    r = torch.tensor(rewards, dtype=torch.float32)
    k = r.shape[0]

    if k < 2:
        return torch.zeros_like(r)

    if torch.allclose(r, r[0]):
        return torch.zeros_like(r)

    log_k = math.log(k)

    def kl_hat(beta_scalar: float) -> float:
        beta_tensor = r.new_tensor(beta_scalar)
        logits = beta_tensor * (r - r.max())
        log_q = logits - torch.logsumexp(logits, dim=0)
        q = torch.exp(log_q)
        kl = (q * (log_q + log_k)).sum()
        return float(kl.item())

    lo = 0.0
    hi = 1.0

    beta = None

    if kl_hat(hi) < delta:
        while hi < beta_max and kl_hat(hi) < delta:
            hi *= 2.0

        if kl_hat(hi) < delta:
            beta = r.new_tensor(hi)

    if beta is None:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if kl_hat(mid) < delta:
                lo = mid
            else:
                hi = mid
        beta = r.new_tensor(hi)

    stable_r = r - r.max()
    e = torch.exp(beta * stable_r)

    # Leave-one-out denominator.
    # For each i, use mean of exp(beta * r_j) where j != i.
    z = (e.sum() - e) / (k - 1)

    advantages = e / (z + eps) - 1.0

    # Safety against rare numerical issues.
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)

    return advantages


def encode_prompt_and_response(
    tokenizer,
    prompt_text: str,
    response_text: str,
    max_length: int = 4096,
):
    """
    Tokenize prompt + response together.

    We train only on the response tokens.
    Prompt tokens are masked out with label = -100.
    """

    full_text = prompt_text + "\n" + response_text

    prompt_encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    full_encoded = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]

    labels = input_ids.clone()

    prompt_length = len(prompt_encoded["input_ids"])
    prompt_length = min(prompt_length, labels.shape[1])

    labels[:, :prompt_length] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def compute_response_ce_loss(
    model,
    tokenizer,
    prompt_text: str,
    response_text: str,
    max_length: int = 4096,
) -> Optional[torch.Tensor]:
    """
    Compute cross-entropy loss only on the generated response.

    This returns normal language-modeling CE:
        CE = - log probability of response tokens

    Later we multiply CE by advantage:
        loss_i = advantage_i * CE_i

    If advantage is positive:
        minimizing loss lowers CE, so model becomes more likely to generate this response.

    If advantage is negative:
        minimizing loss raises CE, so model becomes less likely to generate this response.
    """

    batch = encode_prompt_and_response(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        response_text=response_text,
        max_length=max_length,
    )

    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    if torch.all(labels == -100):
        return None

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    return outputs.loss


def train_lora_on_rollouts(
    model,
    tokenizer,
    optimizer,
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    beta: float = 2.0,
    max_length: int = 4096,
):
    """
    Perform one local LoRA update using entropic adaptive-beta advantages.

    This is closer to original TTT-Discover than reward-weighted SFT.

    Important:
    - This is still not the full original importance-sampling loss.
    - It does implement the key original advantage idea:
        entropic_adaptive_beta
    - Full importance sampling would also require storing generation-time token logprobs.
    """

    if not (len(prompts) == len(responses) == len(rewards)):
        raise ValueError("prompts, responses, and rewards must have the same length")

    if len(prompts) == 0:
        return {
            "ok": False,
            "loss": None,
            "message": "No rollouts provided.",
        }

    model.train()
    optimizer.zero_grad()

    advantages = compute_entropic_adaptive_beta_advantages(rewards)

    # If all rewards are the same, there is no learning signal.
    if torch.allclose(advantages, torch.zeros_like(advantages)):
        optimizer.zero_grad()
        return {
            "ok": False,
            "loss": None,
            "message": "All rewards are equal or only one rollout was available; advantages are zero.",
            "rewards": rewards,
            "advantages": advantages.detach().cpu().tolist(),
        }

    total_loss = None
    used_count = 0
    per_sample_losses = []

    for prompt_text, response_text, advantage in zip(prompts, responses, advantages):
        ce_loss = compute_response_ce_loss(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            response_text=response_text,
            max_length=max_length,
        )

        if ce_loss is None:
            per_sample_losses.append(None)
            continue

        advantage = advantage.to(ce_loss.device)

        # Policy-gradient style local approximation:
        # CE = -logprob(response)
        # loss = advantage * CE
        #
        # Positive advantage -> reduce CE -> increase probability.
        # Negative advantage -> increase CE -> decrease probability.
        sample_loss = advantage * ce_loss

        if total_loss is None:
            total_loss = sample_loss
        else:
            total_loss = total_loss + sample_loss

        used_count += 1
        per_sample_losses.append(float(ce_loss.detach().cpu()))

    if total_loss is None or used_count == 0:
        optimizer.zero_grad()
        return {
            "ok": False,
            "loss": None,
            "message": "No valid response tokens were available for training.",
            "rewards": rewards,
            "advantages": advantages.detach().cpu().tolist(),
        }

    # Average over used samples for stability.
    total_loss = total_loss / used_count

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {
        "ok": True,
        "loss": float(total_loss.detach().cpu()),
        "used_count": used_count,
        "rewards": rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "per_sample_ce_losses": per_sample_losses,
    }


if __name__ == "__main__":
    print("This file provides local entropic-advantage LoRA training helpers.")
    print("It is closer to TTT-Discover than simple reward-weighted SFT.")
