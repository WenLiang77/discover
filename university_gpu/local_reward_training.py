from typing import List, Optional

import torch


def normalize_rewards_to_weights(rewards: List[float], beta: float = 2.0):
    """
    Convert rewards into training weights.

    Higher reward should get larger weight.
    This is a simplified version of the TTT-Discover idea:
    high-reward generations should influence the model update more.

    Args:
        rewards:
            A list of reward values. Higher is better.
        beta:
            Controls how strongly we emphasize high-reward samples.

    Returns:
        A torch tensor of weights summing to 1.
    """
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    if len(reward_tensor) == 0:
        raise ValueError("rewards cannot be empty")

    if len(reward_tensor) == 1:
        return torch.ones_like(reward_tensor)

    if torch.allclose(reward_tensor, reward_tensor[0]):
        return torch.ones_like(reward_tensor) / len(reward_tensor)

    mean = reward_tensor.mean()
    std = reward_tensor.std().clamp_min(1e-6)

    normalized_rewards = (reward_tensor - mean) / std
    weights = torch.softmax(beta * normalized_rewards, dim=0)

    return weights


def encode_prompt_and_response(
    tokenizer,
    prompt_text: str,
    response_text: str,
    max_length: int = 4096,
):
    """
    Tokenize prompt + response together.

    We want the model to learn from the response only.
    Therefore, prompt tokens will be masked out with label = -100.
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


def compute_response_loss(
    model,
    tokenizer,
    prompt_text: str,
    response_text: str,
    max_length: int = 4096,
) -> Optional[torch.Tensor]:
    """
    Compute language modeling loss only on the generated response.

    The prompt part is ignored by setting its labels to -100.
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
    Perform one tiny reward-weighted LoRA update.

    Args:
        model:
            The local Hugging Face model with LoRA adapters.
        tokenizer:
            Tokenizer for the model.
        optimizer:
            PyTorch optimizer for LoRA parameters.
        prompts:
            Prompts sent to the model.
        responses:
            Generated code responses from the model.
        rewards:
            Evaluator rewards for each response.
        beta:
            Controls how much high-reward samples are emphasized.
        max_length:
            Maximum token length for prompt + response.

    Returns:
        A dictionary containing training status and loss.
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

    weights = normalize_rewards_to_weights(rewards, beta=beta)

    total_loss = None
    used_count = 0
    per_sample_losses = []

    for prompt_text, response_text, weight in zip(prompts, responses, weights):
        loss = compute_response_loss(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            response_text=response_text,
            max_length=max_length,
        )

        if loss is None:
            per_sample_losses.append(None)
            continue

        weight = weight.to(loss.device)
        weighted_loss = weight * loss

        if total_loss is None:
            total_loss = weighted_loss
        else:
            total_loss = total_loss + weighted_loss

        used_count += 1
        per_sample_losses.append(float(loss.detach().cpu()))

    if total_loss is None or used_count == 0:
        optimizer.zero_grad()
        return {
            "ok": False,
            "loss": None,
            "message": "No valid response tokens were available for training.",
        }

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {
        "ok": True,
        "loss": float(total_loss.detach().cpu()),
        "used_count": used_count,
        "weights": weights.detach().cpu().tolist(),
        "per_sample_losses": per_sample_losses,
    }


if __name__ == "__main__":
    print("This file provides local reward-weighted LoRA training helpers.")
    print("It is not meant to run the full experiment by itself.")