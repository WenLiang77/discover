import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


DEFAULT_MODEL_NAME = os.environ.get(
    "LOCAL_TTT_MODEL",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
)


def get_compute_dtype():
    """
    Choose a safe compute dtype.

    On modern NVIDIA GPUs such as A100/H100, bf16 is usually supported.
    If bf16 is not supported, use fp16.
    If CUDA is not available, use fp32 on CPU.
    """
    if not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def get_lora_target_modules(model_name: str):
    """
    Tell PEFT which linear layers should receive LoRA adapters.

    For Qwen / Llama style decoder-only models, these module names are common:
    - q_proj, k_proj, v_proj, o_proj: attention projections
    - gate_proj, up_proj, down_proj: MLP layers

    If the model does not have some of these names, PEFT may raise an error.
    For Qwen2.5-Coder this list should be suitable.
    """
    name = model_name.lower()

    if "qwen" in name or "llama" in name or "mistral" in name:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def load_tokenizer_and_lora_model(
    model_name: str = DEFAULT_MODEL_NAME,
    lora_rank: int = 8,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
):
    """
    Load a Hugging Face causal language model and attach LoRA adapters.

    This is the local university-GPU replacement for the Tinker training client.

    Returns:
        tokenizer
        model with LoRA adapters
    """
    print("=" * 80)
    print("Loading tokenizer and local LoRA model")
    print("=" * 80)
    print("Model name:", model_name)
    print("LoRA rank:", lora_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_compute_dtype()
    print("Compute dtype:", dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Important for training with LoRA.
    # Some models use cache during generation, but training with gradient checkpointing /
    # adapters is safer with cache disabled.
    model.config.use_cache = False

    if lora_alpha is None:
        lora_alpha = 2 * lora_rank

    target_modules = get_lora_target_modules(model_name)
    print("LoRA target modules:", target_modules)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    return tokenizer, model


def make_optimizer(model, learning_rate: float = 1e-4):
    """
    Create an optimizer for LoRA trainable parameters only.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("=" * 80)
    print("Creating optimizer")
    print("=" * 80)
    print("Number of trainable parameter tensors:", len(trainable_params))
    print("Learning rate:", learning_rate)

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    return optimizer


def quick_model_check():
    """
    A tiny test function.

    This only checks whether the model and LoRA adapter can be loaded.
    It does not run the denoising experiment.
    """
    tokenizer, model = load_tokenizer_and_lora_model()
    optimizer = make_optimizer(model)

    print("=" * 80)
    print("Quick check finished")
    print("=" * 80)
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model class:", model.__class__.__name__)
    print("Optimizer class:", optimizer.__class__.__name__)


if __name__ == "__main__":
    quick_model_check()

# ============================================================
# Original-ish TTT policy-gradient LoRA trainer
# This overrides the earlier train_lora_on_rollouts definition.
# Main differences from the simple local trainer:
#   1. group-based entropic advantage
#   2. remove constant-reward groups
#   3. old-policy logprob for importance-sampling style update
#   4. optional KL proxy against base model with adapters disabled
# ============================================================

def _ttt_build_batch(tokenizer, prompts, responses, max_length, device):
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id
    input_ids_list = []
    labels_list = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        if len(response_ids) == 0:
            response_ids = [tokenizer.eos_token_id]

        # Keep as much response/code as possible, and truncate prompt from the left.
        if len(prompt_ids) + len(response_ids) > max_length:
            response_budget = min(len(response_ids), max(1, int(max_length * 0.75)))
            prompt_budget = max_length - response_budget
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            response_ids = response_ids[:response_budget]

        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        if len(ids) < 2:
            continue

        input_ids_list.append(ids)
        labels_list.append(labels)

    if not input_ids_list:
        return None

    batch_len = max(len(x) for x in input_ids_list)

    input_ids = []
    attention_mask = []
    labels = []

    for ids, lab in zip(input_ids_list, labels_list):
        pad_len = batch_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(lab + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _ttt_sequence_mean_logprobs(model, batch):
    import torch
    import torch.nn.functional as F

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )

    logits = outputs.logits
    labels = batch["labels"]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab = shift_logits.size(-1)

    token_nll = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)

    mask = shift_labels.ne(-100)
    token_logp = -token_nll * mask

    denom = mask.sum(dim=1).clamp_min(1)
    seq_mean_logp = token_logp.sum(dim=1) / denom

    return seq_mean_logp


def _ttt_collect_logprobs(model, tokenizer, prompts, responses, max_length, microbatch_size, device, disable_adapter=False):
    import torch

    logps = []

    ctx = None
    if disable_adapter and hasattr(model, "disable_adapter"):
        ctx = model.disable_adapter()

    try:
        if ctx is not None:
            ctx.__enter__()

        model.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), microbatch_size):
                p = prompts[i:i + microbatch_size]
                r = responses[i:i + microbatch_size]
                batch = _ttt_build_batch(tokenizer, p, r, max_length, device)
                if batch is None:
                    continue
                lp = _ttt_sequence_mean_logprobs(model, batch)
                logps.append(lp.detach().cpu())

                del batch, lp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    if not logps:
        return torch.empty(0)

    return torch.cat(logps, dim=0)


def _ttt_compute_group_advantages(rewards, group_size, beta, remove_constant_groups=True):
    import torch

    rewards_t = torch.tensor([float(x) for x in rewards], dtype=torch.float32)
    n = len(rewards)

    advantages = torch.zeros(n, dtype=torch.float32)
    keep = torch.zeros(n, dtype=torch.bool)

    skipped_groups = 0
    used_groups = 0

    for start in range(0, n, group_size):
        end = min(start + group_size, n)
        idx = list(range(start, end))

        if len(idx) <= 1:
            skipped_groups += 1
            continue

        r = rewards_t[idx]

        if remove_constant_groups and float(r.max() - r.min()) < 1e-8:
            skipped_groups += 1
            continue

        # Entropic-style group advantage:
        # Softmax over rewards minus uniform baseline.
        centred = r - r.max()
        probs = torch.softmax(beta * centred, dim=0)
        uniform = torch.full_like(probs, 1.0 / len(idx))
        adv = probs - uniform

        # Normalize advantage scale to avoid one huge reward dominating.
        scale = adv.abs().mean().clamp_min(1e-6)
        adv = (adv / scale).clamp(-5.0, 5.0)

        advantages[idx] = adv
        keep[idx] = True
        used_groups += 1

    return advantages, keep, used_groups, skipped_groups


def train_lora_on_rollouts(
    model,
    tokenizer,
    optimizer,
    prompts,
    responses,
    rewards,
    beta=2.0,
    max_length=1024,
    **kwargs,
):
    """
    Original-ish local TTT LoRA update.

    This is not the official Tinker backend, but it is closer to the paper-style
    update than raw reward-weighted SFT:
      - group advantage instead of raw reward
      - remove constant reward groups
      - importance-sampling ratio using old logprobs
      - optional KL proxy to the base model
    """
    import os
    import math
    import torch

    if not prompts:
        return {
            "ok": False,
            "skipped": True,
            "reason": "No rollout prompts.",
        }

    device = next(model.parameters()).device

    group_size = int(os.environ.get("LOCAL_TTT_GROUP_SIZE", str(len(prompts))))
    microbatch_size = int(os.environ.get("LOCAL_TTT_TRAIN_MICROBATCH", "1"))
    policy_epochs = int(os.environ.get("LOCAL_TTT_POLICY_EPOCHS", "1"))
    clip_ratio = float(os.environ.get("LOCAL_TTT_CLIP_RATIO", "0.2"))
    kl_coef = float(os.environ.get("LOCAL_TTT_KL_COEF", "0.01"))
    remove_constant = os.environ.get("LOCAL_TTT_REMOVE_CONSTANT_GROUPS", "1") != "0"

    advantages, keep_mask, used_groups, skipped_groups = _ttt_compute_group_advantages(
        rewards=rewards,
        group_size=group_size,
        beta=beta,
        remove_constant_groups=remove_constant,
    )

    keep_indices = [i for i, keep in enumerate(keep_mask.tolist()) if keep]

    if not keep_indices:
        return {
            "ok": True,
            "skipped": True,
            "reason": "All reward groups were constant or too small.",
            "num_rollouts": len(prompts),
            "used_groups": used_groups,
            "skipped_groups": skipped_groups,
            "mean_reward": float(sum(float(x) for x in rewards) / max(1, len(rewards))),
        }

    train_prompts = [prompts[i] for i in keep_indices]
    train_responses = [responses[i] for i in keep_indices]
    train_advantages = advantages[keep_indices].to(device)

    # Old policy logprobs are computed before the LoRA update.
    old_logps = _ttt_collect_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompts=train_prompts,
        responses=train_responses,
        max_length=max_length,
        microbatch_size=microbatch_size,
        device=device,
        disable_adapter=False,
    ).to(device)

    if old_logps.numel() != len(train_prompts):
        return {
            "ok": False,
            "skipped": True,
            "reason": "Could not compute old logprobs for all examples.",
            "num_old_logps": int(old_logps.numel()),
            "num_examples": len(train_prompts),
        }

    # Reference/base logprobs for a cheap KL proxy.
    # If disable_adapter is unavailable, fall back to old_logps.
    if kl_coef > 0:
        try:
            ref_logps = _ttt_collect_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompts=train_prompts,
                responses=train_responses,
                max_length=max_length,
                microbatch_size=microbatch_size,
                device=device,
                disable_adapter=True,
            ).to(device)

            if ref_logps.numel() != len(train_prompts):
                ref_logps = old_logps.detach()
        except Exception as e:
            print("Warning: KL reference logprob failed; falling back to old policy logprobs:", repr(e))
            ref_logps = old_logps.detach()
    else:
        ref_logps = old_logps.detach()

    model.train()

    total_loss_value = 0.0
    total_pg_value = 0.0
    total_kl_value = 0.0
    update_steps = 0

    for epoch in range(policy_epochs):
        optimizer.zero_grad(set_to_none=True)

        num_microbatches = math.ceil(len(train_prompts) / microbatch_size)

        for mb_start in range(0, len(train_prompts), microbatch_size):
            mb_end = min(mb_start + microbatch_size, len(train_prompts))

            mb_prompts = train_prompts[mb_start:mb_end]
            mb_responses = train_responses[mb_start:mb_end]

            batch = _ttt_build_batch(
                tokenizer=tokenizer,
                prompts=mb_prompts,
                responses=mb_responses,
                max_length=max_length,
                device=device,
            )

            if batch is None:
                continue

            curr_logps = _ttt_sequence_mean_logprobs(model, batch)

            old_mb = old_logps[mb_start:mb_end].detach()
            ref_mb = ref_logps[mb_start:mb_end].detach()
            adv_mb = train_advantages[mb_start:mb_end].detach()

            log_ratio = (curr_logps - old_mb).clamp(-10.0, 10.0)
            ratio = torch.exp(log_ratio)

            if clip_ratio > 0:
                ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)

            pg_loss = -(ratio * adv_mb).mean()

            # KL proxy: keep adapter policy close to base/reference behavior.
            # This is cheaper than full token-distribution KL.
            kl_proxy = ((curr_logps - ref_mb) ** 2).mean()

            loss = pg_loss + kl_coef * kl_proxy
            loss = loss / max(1, num_microbatches)
            loss.backward()

            total_loss_value += float(loss.detach().cpu())
            total_pg_value += float(pg_loss.detach().cpu())
            total_kl_value += float(kl_proxy.detach().cpu())
            update_steps += 1

            del batch, curr_logps, old_mb, ref_mb, adv_mb, log_ratio, ratio, pg_loss, kl_proxy, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {
        "ok": True,
        "skipped": False,
        "trainer": "originalish_group_advantage_is_kl",
        "num_rollouts": len(prompts),
        "num_train_examples": len(train_prompts),
        "group_size": group_size,
        "used_groups": used_groups,
        "skipped_groups": skipped_groups,
        "policy_epochs": policy_epochs,
        "microbatch_size": microbatch_size,
        "beta": beta,
        "clip_ratio": clip_ratio,
        "kl_coef": kl_coef,
        "mean_reward": float(sum(float(x) for x in rewards) / max(1, len(rewards))),
        "mean_abs_advantage": float(train_advantages.abs().mean().detach().cpu()),
        "loss": total_loss_value / max(1, update_steps),
        "pg_loss": total_pg_value / max(1, update_steps),
        "kl_proxy": total_kl_value / max(1, update_steps),
    }

