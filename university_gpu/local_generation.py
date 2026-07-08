import re
import torch


def build_denoising_prompt(current_code: str, history=None):
    """
    Build the denoising prompt.

    This function first tries to use the official Section 4.4 denoising prompt:
    - examples.denoising.prompt.SYSTEM_PROMPT
    - examples.denoising.utils.EVALUATE_MSE_FUNC
    - examples.denoising.utils.EVALUATE_POISSON_FUNC

    If those imports fail, it falls back to a simpler local prompt.
    """
    if history is None:
        history = []

    try:
        return build_official_denoising_prompt(
            current_code=current_code,
            history=history,
        )
    except Exception as error:
        print("Warning: failed to build official denoising prompt.")
        print("Falling back to simplified local prompt.")
        print("Error:", repr(error))

        return build_simple_denoising_prompt(
            current_code=current_code,
            history=history,
        )


def build_official_denoising_prompt(current_code: str, history=None):
    """
    Build a prompt that is much closer to the official examples/denoising/env.py logic.

    Official denoising prompt components:
    - SYSTEM_PROMPT from examples.denoising.prompt
    - evaluate_mse source code
    - evaluate_poisson source code
    - current implementation
    - current / previous metrics
    """
    if history is None:
        history = []

    from examples.denoising.prompt import SYSTEM_PROMPT
    from examples.denoising.utils import EVALUATE_MSE_FUNC, EVALUATE_POISSON_FUNC

    prompt = SYSTEM_PROMPT

    prompt = prompt.replace("<<>>", EVALUATE_MSE_FUNC, 1)
    prompt = prompt.replace("<<>>", EVALUATE_POISSON_FUNC, 1)

    metrics_text = build_history_text(history)

    clean_code = clean_generated_code_block(current_code)

    prompt = prompt + """

CURRENT_IMPLEMENTATION_BEGIN

""" + clean_code + """

CURRENT_IMPLEMENTATION_END

You are iteratively improving the denoising algorithm.

Previous evaluation results:
""" + metrics_text + """

Now write an improved version of magic_denoise.

Strict code requirements:
- Return only executable Python code.
- Do not include explanations, markdown fences, or comments outside the code.
- Do not repeat CURRENT_IMPLEMENTATION_BEGIN, CURRENT_IMPLEMENTATION_END, or previous history text.
- The code must be self-contained.
- Include all required imports explicitly in the generated code.
- At minimum, include `import numpy as np` if you use `np`.
- The function signature must be exactly:
  def magic_denoise(X, **kwargs):
- Do not introduce required positional arguments other than X.
- Do not use undefined variables or undefined functions.
- If you use sklearn, scipy, scanpy, scprep, or graphtools objects, import them explicitly.
- Common tools must be imported explicitly if used: `from sklearn.decomposition import PCA, TruncatedSVD`, `from sklearn.preprocessing import StandardScaler, normalize`, `from sklearn.neighbors import NearestNeighbors`, `from scipy.optimize import minimize`, `from scipy.stats import poisson`, and `from scipy.spatial import distance_matrix`.
- Do not assume helper names such as PCA, StandardScaler, minimize, poisson, distance_matrix, normalize, cdist, or NearestNeighbors already exist.
- Prefer robust improvements to the current implementation rather than rewriting everything from scratch.
- Avoid algorithms that return NaN, inf, negative values, or outputs with unstable scale.
- Passing the Poisson constraint is required; do not optimize MSE by producing unrealistic count values.
- The output must be a numpy array with the same shape as X.
- The output must be finite and non-negative.
- Avoid file I/O, network access, and external datasets.
- Keep the implementation reasonably simple and robust.
"""

    return prompt.strip()


def build_simple_denoising_prompt(current_code: str, history=None):
    """
    Fallback prompt.

    This is only used if the official denoising prompt cannot be imported.
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
The code must be self-contained and include all required imports.
Do not use undefined names such as PCA, StandardScaler, minimize, poisson, distance_matrix, normalize, cdist, or NearestNeighbors unless you import them explicitly.
The function must return finite, non-negative output with the same shape as X.
Prefer simple, stable numpy/scipy/sklearn/scprep/graphtools code.
- The code must define a function called magic_denoise.
- Try to improve MSE while keeping Poisson loss reasonable.

Previous evaluation results:
{history_text}

CURRENT_IMPLEMENTATION_BEGIN

{current_code}

CURRENT_IMPLEMENTATION_END

Now write an improved version of magic_denoise.
Return only Python code.
"""
    return prompt.strip()


def build_history_text(history):
    """
    Convert previous rollout records into a short text summary.

    Diversity-aware version:
    - repeated valid behavior is not shown as a success to imitate
    - repeated metric plateaus are listed as behaviors to avoid
    """
    if not history:
        return "No previous generated results yet."

    duplicate_valid = [
        item for item in history
        if item.get("ok") and item.get("duplicate_valid_behavior")
    ]

    avoid_mse_values = []
    avoid_metric_signatures = set()

    for item in duplicate_valid:
        mse = item.get("mse")
        metrics = item.get("metrics", {}) or {}
        poisson_norm = metrics.get("poisson_normalized")

        if mse is not None:
            mse_sig = round(float(mse), 12)
            avoid_mse_values.append(mse_sig)

            if poisson_norm is not None:
                avoid_metric_signatures.add((mse_sig, round(float(poisson_norm), 12)))

    avoid_mse_values = sorted(set(avoid_mse_values))

    successful = []
    for item in history:
        if not item.get("ok"):
            continue

        mse = item.get("mse")
        metrics = item.get("metrics", {}) or {}
        poisson_norm = metrics.get("poisson_normalized")

        is_avoided_plateau = False
        if mse is not None:
            mse_sig = round(float(mse), 12)
            if mse_sig in avoid_mse_values:
                is_avoided_plateau = True

            if poisson_norm is not None:
                sig = (mse_sig, round(float(poisson_norm), 12))
                if sig in avoid_metric_signatures:
                    is_avoided_plateau = True

        if item.get("duplicate_valid_behavior"):
            is_avoided_plateau = True

        if not is_avoided_plateau:
            successful.append(item)

    lines = []

    if avoid_mse_values:
        avoid_text = ", ".join(str(x) for x in avoid_mse_values[:5])
        lines.append(
            "Avoid repeating already-discovered plateau behaviors. "
            f"In particular, do not generate another implementation likely to obtain these MSE values again: {avoid_text}. "
            "Try a substantially different denoising strategy instead."
        )

    if not successful:
        if lines:
            lines.append(
                "No non-duplicate successful improvement is available yet. "
                "Explore a different robust algorithm while keeping the output finite, non-negative, and Poisson-valid."
            )
            return "\n".join(lines)

        return "Previous attempts failed. Please produce simple, valid, robust Python code."

    successful = sorted(
        successful,
        key=lambda item: item.get("mse", float("inf")),
    )

    top = successful[:3]

    if lines:
        lines.append("Non-duplicate successful results so far:")

    for i, item in enumerate(top, start=1):
        metrics = item.get("metrics", {})

        mse_normalized = metrics.get("mse_normalized")
        poisson_normalized = metrics.get("poisson_normalized")

        line = (
            f"{i}. mse={item.get('mse')}, "
            f"poisson={item.get('poisson')}, "
            f"reward={item.get('reward')}"
        )

        if mse_normalized is not None:
            line += f", mse_normalized={mse_normalized}"

        if poisson_normalized is not None:
            line += f", poisson_normalized={poisson_normalized}"

        lines.append(line)

    return "\n".join(lines)


def clean_generated_code_block(text: str):
    """
    Remove markdown code fences if they are present.
    """
    text = text.strip()

    if text.startswith("```python"):
        text = text[len("```python"):].strip()

    if text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text.strip()


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
        max_length=8192,
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
    print("It now tries to use the official denoising prompt first.")