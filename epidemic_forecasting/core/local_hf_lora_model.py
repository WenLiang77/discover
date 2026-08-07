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
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

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
