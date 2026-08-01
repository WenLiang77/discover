from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and nested containers into JSON-safe objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _normalise_code(code: str) -> str:
    """Normalise formatting before exact-code duplicate detection."""
    code = code.strip()

    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    elif code.startswith("```"):
        code = code[3:].strip()

    if code.endswith("```"):
        code = code[:-3].strip()

    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def code_fingerprint(code: str) -> str:
    """Return a stable SHA-256 fingerprint for candidate code."""
    normalised = _normalise_code(code)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


@dataclass
class SearchState:
    """
    Task-independent state stored in the PUCT search buffer.

    value must be a scalar reward for which larger is better.
    metrics may contain task-specific values such as SMAPE, MAE or RMSE.
    """

    timestep: int = 0
    code: str = ""
    value: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    behavior_signature: str | None = None
    parent_values: list[float] = field(default_factory=list)
    parents: list[dict[str, Any]] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "type": "SearchState",
                "id": self.id,
                "timestep": self.timestep,
                "code": self.code,
                "value": self.value,
                "metrics": self.metrics,
                "observation": self.observation,
                "metadata": self.metadata,
                "behavior_signature": self.behavior_signature,
                "parent_values": self.parent_values,
                "parents": self.parents,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchState":
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            timestep=int(data.get("timestep", 0)),
            code=str(data.get("code", "")),
            value=data.get("value"),
            metrics=dict(data.get("metrics", {}) or {}),
            observation=str(data.get("observation", "")),
            metadata=dict(data.get("metadata", {}) or {}),
            behavior_signature=data.get("behavior_signature"),
            parent_values=list(data.get("parent_values", []) or []),
            parents=list(data.get("parents", []) or []),
        )


class PUCTSampler:
    """
    Task-independent PUCT sampler.

    This class selects promising parent states and stores evaluated children.
    It does not generate code and does not calculate disease-specific metrics.

    For candidate i:

        score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n(i))

    Larger state values are always treated as better.
    """

    def __init__(
        self,
        file_path: str | Path,
        initial_state_factory: Callable[[], SearchState],
        state_type: type[SearchState] = SearchState,
        max_buffer_size: int = 1000,
        batch_size: int = 1,
        puct_c: float = 1.0,
        topk_children: int = 2,
        resume: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be at least 1.")

        self.file_path = Path(file_path)
        self.initial_state_factory = initial_state_factory
        self.state_type = state_type
        self.max_buffer_size = int(max_buffer_size)
        self.batch_size = int(batch_size)
        self.puct_c = float(puct_c)
        self.topk_children = int(topk_children)

        self._states: list[SearchState] = []
        self._initial_states: list[SearchState] = []
        self._last_sampled_states: list[SearchState] = []
        self._last_puct_stats: list[dict[str, float]] = []

        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T = 0
        self._current_step = 0
        self._last_scale = 1.0

        if resume:
            self._load_if_exists()

        if not self._states:
            for _ in range(self.batch_size):
                state = self._make_initial_state()
                self._initial_states.append(state)
                self._states.append(state)
            self._save(step=0)

    def _make_initial_state(self) -> SearchState:
        state = self.initial_state_factory()

        if not isinstance(state, self.state_type):
            raise TypeError(
                "initial_state_factory must return an instance of "
                f"{self.state_type.__name__}."
            )

        return state

    def _sampler_file_for_step(self, step: int) -> Path:
        base = str(self.file_path)
        if base.endswith(".json"):
            base = base[:-5]
        return Path(f"{base}_step_{step:06d}.json")

    def _latest_file(self) -> Path | None:
        parent = self.file_path.parent
        stem = self.file_path.stem
        files = sorted(parent.glob(f"{stem}_step_*.json"))
        return files[-1] if files else None

    def _load_if_exists(self) -> None:
        latest = self._latest_file()
        if latest is None:
            return

        try:
            data = json.loads(latest.read_text(encoding="utf-8"))

            self._states = [
                self.state_type.from_dict(item)
                for item in data.get("states", [])
            ]
            self._initial_states = [
                self.state_type.from_dict(item)
                for item in data.get("initial_states", [])
            ]
            self._n = {
                str(key): int(value)
                for key, value in (data.get("puct_n", {}) or {}).items()
            }
            self._m = {
                str(key): float(value)
                for key, value in (data.get("puct_m", {}) or {}).items()
            }
            self._T = int(data.get("puct_T", 0) or 0)
            self._current_step = int(data.get("step", 0) or 0)

            print(f"Loaded PUCT sampler from {latest}")
        except Exception as error:
            print("Warning: failed to load the previous PUCT sampler.")
            print("Starting with a new search buffer.")
            print("Error:", repr(error))

            self._states = []
            self._initial_states = []
            self._n = {}
            self._m = {}
            self._T = 0
            self._current_step = 0

    def _save(self, step: int) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        path = self._sampler_file_for_step(step)

        data = {
            "step": step,
            "states": [state.to_dict() for state in self._states],
            "initial_states": [
                state.to_dict()
                for state in self._initial_states
            ],
            "puct_n": self._n,
            "puct_m": self._m,
            "puct_T": self._T,
            "settings": {
                "max_buffer_size": self.max_buffer_size,
                "batch_size": self.batch_size,
                "puct_c": self.puct_c,
                "topk_children": self.topk_children,
            },
        }

        temporary_path = Path(str(path) + ".tmp")
        temporary_path.write_text(
            json.dumps(_json_safe(data), indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _state_key(self, state: SearchState) -> str:
        if state.behavior_signature:
            return f"behavior:{state.behavior_signature}"
        if state.code:
            return f"code:{code_fingerprint(state.code)}"
        return f"id:{state.id}"

    @staticmethod
    def _finite_value(value: Any) -> float | None:
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        return number if math.isfinite(number) else None

    @staticmethod
    def _compute_scale(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            return 1.0
        return float(max(np.max(finite) - np.min(finite), 1e-6))

    @staticmethod
    def _compute_prior(values: np.ndarray) -> np.ndarray:
        count = len(values)
        if count == 0:
            return np.array([], dtype=np.float64)

        if not np.isfinite(values).any():
            return np.ones(count, dtype=np.float64) / count

        safe_values = np.where(np.isfinite(values), values, -1e30)
        ranks = np.argsort(
            np.argsort(-safe_values, kind="stable"),
            kind="stable",
        )
        weights = (count - ranks).astype(np.float64)
        return weights / weights.sum()

    @staticmethod
    def _lineage_ids(state: SearchState) -> list[str]:
        lineage = [state.id]
        lineage.extend(
            str(parent["id"])
            for parent in state.parents
            if parent.get("id")
        )
        return lineage

    @staticmethod
    def _set_parent_info(
        child: SearchState,
        parent: SearchState,
    ) -> None:
        child.parent_values = (
            [float(parent.value)] + list(parent.parent_values)
            if parent.value is not None
            else list(parent.parent_values)
        )
        child.parents = [
            {
                "id": parent.id,
                "timestep": parent.timestep,
            }
        ] + list(parent.parents)

    def sample_states(self, num_states: int) -> list[SearchState]:
        if num_states < 1:
            raise ValueError("num_states must be at least 1.")

        candidates = list(self._states)

        if not candidates:
            picked = [
                self._make_initial_state()
                for _ in range(num_states)
            ]
            self._last_sampled_states = picked
            self._last_puct_stats = []
            return picked

        values = np.array(
            [
                float(state.value)
                if self._finite_value(state.value) is not None
                else float("-inf")
                for state in candidates
            ],
            dtype=np.float64,
        )

        scale = self._compute_scale(values)
        prior = self._compute_prior(values)
        sqrt_t = math.sqrt(1.0 + self._T)
        self._last_scale = scale

        scored = []

        for index, state in enumerate(candidates):
            own_value = self._finite_value(state.value)
            backed_up_value = self._finite_value(self._m.get(state.id))

            available_values = [
                value
                for value in (own_value, backed_up_value)
                if value is not None
            ]
            q_value = max(available_values) if available_values else 0.0

            visits = self._n.get(state.id, 0)
            bonus = (
                self.puct_c
                * scale
                * float(prior[index])
                * sqrt_t
                / (1.0 + visits)
            )
            score = q_value + bonus

            scored.append(
                {
                    "state": state,
                    "value": own_value,
                    "visits": visits,
                    "Q": q_value,
                    "P": float(prior[index]),
                    "bonus": bonus,
                    "score": score,
                }
            )

        scored.sort(
            key=lambda item: (
                item["score"],
                item["value"]
                if item["value"] is not None
                else float("-inf"),
            ),
            reverse=True,
        )

        selected = scored[: min(num_states, len(scored))]
        self._last_sampled_states = [
            item["state"]
            for item in selected
        ]
        self._last_puct_stats = [
            {
                "n": float(item["visits"]),
                "Q": float(item["Q"]),
                "P": float(item["P"]),
                "bonus": float(item["bonus"]),
                "score": float(item["score"]),
            }
            for item in selected
        ]

        return list(self._last_sampled_states)

    def update_states(
        self,
        states: list[SearchState],
        parent_states: list[SearchState],
        save: bool = True,
        step: int | None = None,
    ) -> None:
        if not states:
            return

        if len(states) != len(parent_states):
            raise ValueError(
                "states and parent_states must have the same length."
            )

        valid_pairs: list[tuple[SearchState, SearchState]] = []

        for child, parent in zip(states, parent_states):
            child_value = self._finite_value(child.value)
            if child_value is None:
                continue

            self._set_parent_info(child, parent)
            valid_pairs.append((child, parent))

            for ancestor_id in self._lineage_ids(parent):
                self._n[ancestor_id] = self._n.get(ancestor_id, 0) + 1
                self._m[ancestor_id] = max(
                    self._m.get(ancestor_id, float("-inf")),
                    child_value,
                )

            self._T += 1

        if self.topk_children > 0:
            grouped: dict[str, list[tuple[SearchState, SearchState]]] = {}

            for child, parent in valid_pairs:
                grouped.setdefault(parent.id, []).append((child, parent))

            valid_pairs = []

            for pairs in grouped.values():
                pairs.sort(
                    key=lambda pair: float(pair[0].value),
                    reverse=True,
                )
                valid_pairs.extend(pairs[: self.topk_children])

        existing_keys = {
            self._state_key(state)
            for state in self._states
        }

        new_states = []

        for child, _parent in valid_pairs:
            key = self._state_key(child)

            if key in existing_keys:
                continue

            new_states.append(child)
            existing_keys.add(key)

        self._states.extend(new_states)

        if len(self._states) > self.max_buffer_size:
            self._states.sort(
                key=lambda state: (
                    self._finite_value(state.value)
                    if self._finite_value(state.value) is not None
                    else float("-inf")
                ),
                reverse=True,
            )
            self._states = self._states[: self.max_buffer_size]

        if save:
            self.flush(step=step)

    def record_failed_rollout(
        self,
        parent: SearchState,
        save: bool = False,
        step: int | None = None,
    ) -> None:
        for ancestor_id in self._lineage_ids(parent):
            self._n[ancestor_id] = self._n.get(ancestor_id, 0) + 1

        self._T += 1

        if save:
            self.flush(step=step)

    def flush(self, step: int | None = None) -> None:
        if step is not None:
            self._current_step = int(step)

        self._save(step=self._current_step)

    def best_states(self, limit: int = 10) -> list[SearchState]:
        valid_states = [
            state
            for state in self._states
            if self._finite_value(state.value) is not None
        ]
        valid_states.sort(
            key=lambda state: float(state.value),
            reverse=True,
        )
        return valid_states[: max(0, limit)]

    def get_sample_stats(self) -> dict[str, float]:
        values = [
            float(state.value)
            for state in self._states
            if self._finite_value(state.value) is not None
        ]
        sampled_values = [
            float(state.value)
            for state in self._last_sampled_states
            if self._finite_value(state.value) is not None
        ]

        def summarise(
            prefix: str,
            numbers: list[float],
        ) -> dict[str, float]:
            if not numbers:
                return {}

            array = np.asarray(numbers, dtype=np.float64)
            return {
                f"{prefix}/mean": float(array.mean()),
                f"{prefix}/std": float(array.std()),
                f"{prefix}/min": float(array.min()),
                f"{prefix}/max": float(array.max()),
            }

        output = {
            "puct/buffer_size": float(len(self._states)),
            "puct/sampled_size": float(len(self._last_sampled_states)),
            "puct/T": float(self._T),
            "puct/scale_last": float(self._last_scale),
        }
        output.update(summarise("puct/buffer_value", values))
        output.update(summarise("puct/sampled_value", sampled_values))

        if self._last_puct_stats:
            selected_scores = [
                item["score"]
                for item in self._last_puct_stats
            ]
            output.update(
                summarise("puct/selected_score", selected_scores)
            )

        return output


if __name__ == "__main__":
    print("This module provides task-independent PUCT search utilities.")
