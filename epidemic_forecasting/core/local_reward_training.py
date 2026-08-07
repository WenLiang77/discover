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
    Tokenize prompt and response separately.

    Training loss is computed only on response tokens.

    When the combined sequence is longer than max_length, preserve
    response tokens first and truncate the OLD part of the prompt.
    This prevents a long TTT prompt from removing the generated
    response completely.

    With max_length=1024, at least roughly half of the available
    sequence is reserved for the response when the response is long.
    """
    if max_length < 4:
        raise ValueError("max_length must be at least 4.")

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
    )["input_ids"]

    separator_ids = tokenizer(
        "\n",
        add_special_tokens=False,
    )["input_ids"]

    if not response_ids:
        input_ids = torch.tensor(
            [prompt_ids[-max_length:]],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    separator_length = len(separator_ids)

    # Guarantee substantial room for generated response tokens.
    # For max_length=1024 this reserves up to 512 response tokens.
    minimum_response_tokens = min(
        len(response_ids),
        max_length // 2,
    )

    prompt_budget = max(
        0,
        max_length
        - separator_length
        - minimum_response_tokens,
    )

    # Keep the most recent part of the prompt.  In TTT this contains
    # the current model-generated candidate, recent metrics, and the
    # final improvement instruction.
    if prompt_budget > 0:
        kept_prompt_ids = prompt_ids[-prompt_budget:]
    else:
        kept_prompt_ids = []

    # Whatever space remains is given to the response.  If the prompt
    # is short, more than half of the sequence can therefore be used
    # for the response.
    response_budget = (
        max_length
        - len(kept_prompt_ids)
        - separator_length
    )

    kept_response_ids = response_ids[:response_budget]

    combined_ids = (
        kept_prompt_ids
        + separator_ids
        + kept_response_ids
    )

    prompt_and_separator_length = (
        len(kept_prompt_ids)
        + separator_length
    )

    input_ids = torch.tensor(
        [combined_ids],
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()

    # Ignore prompt tokens.  Only generated response tokens contribute
    # to the language-model training loss.
    labels[:, :prompt_and_separator_length] = -100

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

    # Accumulate gradients rollout-by-rollout instead of keeping all
    # computation graphs alive until one final backward pass.
    #
    # This is mathematically equivalent to backpropagating the mean loss:
    # first accumulate the sum of gradients, then divide the gradients by
    # the number of valid training samples before optimizer.step().
    total_loss_value = 0.0
    used_count = 0
    per_sample_losses = []

    for prompt_text, response_text, advantage in zip(
        prompts,
        responses,
        advantages,
    ):
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
        sample_loss = advantage * ce_loss

        # Backpropagate immediately so this rollout's computation graph
        # can be released before the next rollout is processed.
        sample_loss.backward()

        total_loss_value += float(sample_loss.detach().cpu())
        used_count += 1
        per_sample_losses.append(
            float(ce_loss.detach().cpu())
        )

        del sample_loss
        del ce_loss

    if used_count == 0:
        optimizer.zero_grad(set_to_none=True)
        return {
            "ok": False,
            "loss": None,
            "message": (
                "No valid response tokens were available for training."
            ),
            "rewards": rewards,
            "advantages": advantages.detach().cpu().tolist(),
        }

    # The old implementation used:
    #
    #     mean_loss = sum(sample_losses) / used_count
    #     mean_loss.backward()
    #
    # We accumulated the summed gradients above, so divide them here
    # before optimizer.step() to preserve the same average-gradient scale.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(used_count)

    average_loss_value = total_loss_value / used_count

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "ok": True,
        "loss": average_loss_value,
        "used_count": used_count,
        "rewards": rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "per_sample_ce_losses": per_sample_losses,
    }


if __name__ == "__main__":
    print("This file provides local entropic-advantage LoRA training helpers.")
    print("It is closer to TTT-Discover than simple reward-weighted SFT.")
