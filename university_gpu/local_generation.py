import re
import torch


def build_denoising_prompt(current_code: str, history=None):
    """
    Build a prompt for the local LLM.

    The prompt asks the model to improve the current magic_denoise function.
    It also gives the model previous good results if available.
    """
    if history is None:
        history = []

    history_text = build_history_text(history)

    prompt = f"""
You are improving a single-cell RNA-seq denoising algorithm.

Your task is to write an improved Python function named magic_denoise.

Required function signature:

def magic_denoise(X, **kwargs):
    ...

Input:
- X is a numpy array with shape (n_cells, n_genes).
- X contains noisy single-cell RNA-seq count data.

Output:
- Return a numpy array with the same shape as X.
- The output should contain denoised expression values.
- The output must be finite and non-negative.

Allowed libraries:
- numpy
- scipy
- sklearn
- scanpy
- scprep
- graphtools

Important rules:
- Do not use file I/O.
- Do not use network access.
- Do not print explanations.
- Return only Python code.
- The code must define a function called magic_denoise.
- Try to improve MSE while keeping Poisson loss reasonable.

Best previous results:
{history_text}

Current implementation starts here:

{current_code}

Current implementation ends here.

Now write an improved version of magic_denoise.
Return only Python code.
"""
    return prompt.strip()


def build_history_text(history):
    """
    Convert previous rollout records into a short text summary.

    history is expected to be a list of dictionaries.
    Each dictionary may contain:
    - ok
    - mse
    - poisson
    - reward
    """
    if not history:
        return "No previous generated results yet."

    successful = [item for item in history if item.get("ok")]

    if not successful:
        return "Previous attempts failed. Please produce simple, valid, robust Python code."

    successful = sorted(
        successful,
        key=lambda item: item.get("mse", float("inf")),
    )

    top = successful[:3]

    lines = []
    for i, item in enumerate(top, start=1):
        lines.append(
            f"{i}. mse={item.get('mse')}, "
            f"poisson={item.get('poisson')}, "
            f"reward={item.get('reward')}"
        )

    return "\n".join(lines)


def apply_chat_template_if_available(tokenizer, prompt: str):
    """
    Some instruction-tuned models, such as Qwen-Instruct models,
    prefer chat-style input.

    If the tokenizer provides apply_chat_template, we use it.
    Otherwise, we return the raw prompt.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a careful Python scientific computing assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as error:
            print("Warning: apply_chat_template failed.")
            print("Falling back to raw prompt.")
            print("Error:", repr(error))

    return prompt


def generate_candidate_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """
    Generate one candidate magic_denoise function using the local model.

    Returns a dictionary containing:
    - rendered_prompt: the final prompt sent to the model
    - raw_text: the raw model output
    - code: extracted Python code
    """
    model.eval()

    rendered_prompt = apply_chat_template_if_available(tokenizer, prompt)

    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = encoded["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_length:]

    raw_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    code = extract_magic_denoise_code(raw_text)

    return {
        "rendered_prompt": rendered_prompt,
        "raw_text": raw_text,
        "code": code,
    }


def extract_magic_denoise_code(text: str):
    """
    Extract Python code from the model output.

    The model may return:
    - only code
    - markdown-style code block
    - explanation + code

    This function tries to keep only the magic_denoise function.
    """
    text = text.strip()

    code_block = extract_first_python_code_block(text)
    if code_block is not None:
        text = code_block.strip()

    function_index = text.find("def magic_denoise")
    if function_index != -1:
        text = text[function_index:]

    text = remove_trailing_explanation(text)

    return text.strip()


def extract_first_python_code_block(text: str):
    """
    Extract content from the first markdown code block if present.
    """
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def remove_trailing_explanation(text: str):
    """
    Remove obvious trailing explanation after code.

    This is a simple heuristic. It avoids keeping text such as:
    'Explanation: ...'
    after the generated function.
    """
    markers = [
        "\nExplanation:",
        "\nHere is",
        "\nThis function",
        "\nThe function",
    ]

    cut_positions = []

    for marker in markers:
        index = text.find(marker)
        if index != -1:
            cut_positions.append(index)

    if cut_positions:
        text = text[: min(cut_positions)]

    return text


if __name__ == "__main__":
    print("This file provides local generation helper functions.")
    print("It is not meant to run the full experiment by itself.")