from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def load_prompt_template(prompt_path: str | Path) -> str:
    """Load a task-specific prompt template from disk."""
    path = Path(prompt_path)

    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    return path.read_text(encoding="utf-8").strip()


def clean_generated_code_block(text: str) -> str:
    """Remove surrounding Markdown fences from code."""
    text = text.strip()

    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text.strip()


def build_history_text(history: list[dict[str, Any]] | None) -> str:
    """
    Convert previous task evaluations into a compact, task-independent summary.

    Each history item may contain:
    - ok: whether the candidate was valid
    - reward: scalar reward, where larger is better
    - metrics: dictionary of evaluation metrics
    - error: failure message
    - duplicate_valid_behavior: whether the result duplicated an earlier result
    """
    if not history:
        return "No previous generated results yet."

    valid_results = [
        item for item in history
        if item.get("ok") and item.get("reward") is not None
    ]

    failed_results = [
        item for item in history
        if not item.get("ok")
    ]

    duplicate_count = sum(
        1
        for item in valid_results
        if item.get("duplicate_valid_behavior")
    )

    non_duplicate_results = [
        item for item in valid_results
        if not item.get("duplicate_valid_behavior")
    ]

    lines: list[str] = []

    if duplicate_count:
        lines.append(
            f"{duplicate_count} previous valid result(s) duplicated already "
            "discovered behaviour. Avoid reproducing the same algorithm or "
            "the same metric plateau."
        )

    if non_duplicate_results:
        ranked = sorted(
            non_duplicate_results,
            key=lambda item: float(item.get("reward", float("-inf"))),
            reverse=True,
        )

        lines.append("Best non-duplicate valid results so far:")

        for index, item in enumerate(ranked[:3], start=1):
            reward = item.get("reward")
            metrics = item.get("metrics", {}) or {}

            metric_text = ", ".join(
                f"{name}={value}"
                for name, value in sorted(metrics.items())
                if isinstance(value, (int, float))
            )

            line = f"{index}. reward={reward}"

            if metric_text:
                line += f", {metric_text}"

            lines.append(line)
    else:
        lines.append(
            "No non-duplicate valid improvement has been found yet. "
            "Prefer a simple, robust and substantially different approach."
        )

    if failed_results:
        error_messages = []

        for item in failed_results[-3:]:
            error = item.get("error")
            if error:
                error_messages.append(str(error).replace("\n", " ")[:200])

        if error_messages:
            lines.append("Recent failure reasons:")
            lines.extend(f"- {message}" for message in error_messages)

    return "\n".join(lines)


def build_task_prompt(
    prompt_path: str | Path,
    current_code: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    Build a prompt for any epidemic forecasting task.

    The task prompt should contain:
    - {current_code}
    - {current_metrics}
    """
    template = load_prompt_template(prompt_path)

    if "{current_code}" not in template:
        raise ValueError(
            "Prompt template must contain the placeholder {current_code}."
        )

    if "{current_metrics}" not in template:
        raise ValueError(
            "Prompt template must contain the placeholder {current_metrics}."
        )

    clean_code = clean_generated_code_block(current_code)
    metrics_text = build_history_text(history)

    prompt = template.replace("{current_code}", clean_code)
    prompt = prompt.replace("{current_metrics}", metrics_text)

    return prompt.strip()


def apply_chat_template_if_available(tokenizer, prompt: str) -> str:
    """Render the prompt using the tokenizer chat template when available."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful scientific Python assistant specialising "
                "in epidemic forecasting and time-series modelling."
            ),
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
            print("Falling back to the raw prompt.")
            print("Error:", repr(error))

    return prompt


def extract_first_python_code_block(text: str) -> str | None:
    """Extract the first Markdown Python code block, when present."""
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return None


def remove_trailing_explanation(text: str) -> str:
    """Remove common explanatory text appended after generated code."""
    markers = [
        "\nExplanation:",
        "\nHere is why",
        "\nThis implementation",
        "\nThe implementation",
        "\nNotes:",
    ]

    cut_positions = [
        text.find(marker)
        for marker in markers
        if text.find(marker) != -1
    ]

    if cut_positions:
        text = text[:min(cut_positions)]

    return text.strip()


def extract_candidate_code(text: str, function_name: str) -> str:
    """
    Extract a complete candidate implementation from model output.

    Imports and top-level helper functions are retained.
    """
    text = text.strip()

    code_block = extract_first_python_code_block(text)
    if code_block is not None:
        text = code_block

    function_marker = f"def {function_name}"

    possible_starts = [
        position
        for position in (
            text.find("import "),
            text.find("from "),
            text.find(function_marker),
        )
        if position != -1
    ]

    if possible_starts:
        text = text[min(possible_starts):]

    text = remove_trailing_explanation(text)
    text = clean_generated_code_block(text)

    return text.strip()


def generate_candidate_code(
    model,
    tokenizer,
    prompt: str,
    function_name: str,
    max_new_tokens: int = 1024,
    max_input_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> dict[str, str]:
    """Generate one candidate implementation using the local model."""
    model.eval()

    rendered_prompt = apply_chat_template_if_available(tokenizer, prompt)

    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    device = next(model.parameters()).device
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = encoded["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_length:]

    raw_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    code = extract_candidate_code(
        raw_text,
        function_name=function_name,
    )

    return {
        "rendered_prompt": rendered_prompt,
        "raw_text": raw_text,
        "code": code,
    }


if __name__ == "__main__":
    print("This module provides task-independent code-generation helpers.")
