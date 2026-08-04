from __future__ import annotations

import ast

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


def _contains_target_function(
    tree: ast.Module,
    function_name: str,
) -> bool:
    """Return whether a parsed module contains the required top-level function."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
        for node in tree.body
    )


def _parse_longest_complete_candidate(
    text: str,
    function_name: str,
) -> ast.Module | None:
    """
    Parse the complete response.

    If the response ends with unfinished code or explanatory text, repeatedly
    remove trailing lines until a syntactically complete module containing the
    required function is found.
    """
    try:
        return ast.parse(text)
    except SyntaxError:
        pass

    lines = text.splitlines()

    for end_index in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:end_index]).rstrip()

        if not candidate:
            continue

        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue

        if _contains_target_function(tree, function_name):
            return tree

    return None


def _is_literal_assignment(
    node: ast.Assign | ast.AnnAssign,
) -> bool:
    """Return whether a top-level assignment contains a literal constant."""
    value = node.value

    if value is None:
        return True

    try:
        ast.literal_eval(value)
    except (
        ValueError,
        TypeError,
        SyntaxError,
        MemoryError,
        RecursionError,
    ):
        return False

    return True


def _normalise_candidate_module(
    tree: ast.Module,
    function_name: str,
) -> str:
    """
    Keep only module content accepted by the static validator.

    Imports, helper functions, literal constants and one target forecasting
    function are retained. Example calls, print statements and main blocks are
    discarded.
    """
    target_positions = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]

    # When the model returns multiple versions, the final version is normally
    # its intended answer.
    selected_target_position = (
        target_positions[-1] if target_positions else None
    )

    kept_nodes: list[ast.stmt] = []

    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # A future import is unnecessary in generated forecasting code and
            # is not part of the evaluator import allow-list.
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
            ):
                continue

            kept_nodes.append(node)
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                if index == selected_target_position:
                    kept_nodes.append(node)
            else:
                kept_nodes.append(node)
            continue

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if _is_literal_assignment(node):
                kept_nodes.append(node)
            continue

        # Preserve a module docstring, but discard ordinary top-level
        # expressions such as print(...) and example function calls.
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            kept_nodes.append(node)

    cleaned_tree = ast.Module(
        body=kept_nodes,
        type_ignores=[],
    )
    ast.fix_missing_locations(cleaned_tree)

    return ast.unparse(cleaned_tree).strip()


def extract_candidate_code(text: str, function_name: str) -> str:
    """
    Extract and normalise a complete candidate implementation.

    The returned module is aligned with the evaluator's accepted top-level
    structure while retaining imports, helper functions and the final required
    forecasting function.
    """
    text = text.strip()

    code_block = extract_first_python_code_block(text)
    if code_block is not None:
        text = code_block

    text = clean_generated_code_block(text)
    text = remove_trailing_explanation(text)

    # Remove prose preceding the first genuine Python import or definition.
    start_match = re.search(
        r"(?m)^(?:"
        r"from\s+\S+\s+import\s+.+"
        r"|import\s+.+"
        r"|(?:async\s+)?def\s+\w+\s*\("
        r")",
        text,
    )

    if start_match is not None:
        text = text[start_match.start():]

    tree = _parse_longest_complete_candidate(
        text=text,
        function_name=function_name,
    )

    # Leave irreparable output unchanged so that the evaluator can report the
    # original syntax or missing-function error.
    if tree is None:
        return text.strip()

    return _normalise_candidate_module(
        tree=tree,
        function_name=function_name,
    )


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
