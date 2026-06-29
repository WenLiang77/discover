import json
import math
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_MAGIC_FUNC = r'''
import numpy as np

def magic_denoise(X, **kwargs):
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.maximum(X, 0.0)
    return X
'''


LOCAL_BASELINES = {
    "pancreas": {
        "baseline_mse": 0.304721,
        "baseline_poisson": 0.257575,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.031739,
    }
}


def local_verify_denoising(result) -> bool:
    """
    Local copy of the official denoising validity check.

    It avoids importing examples.denoising.env,
    because that file imports ttt_discover and may require tinker.
    """
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return False

    mse, poisson = result[0], result[1]

    if not np.isfinite(mse) or not np.isfinite(poisson):
        return False

    baseline = LOCAL_BASELINES["pancreas"]

    if poisson < baseline["perfect_poisson"]:
        return False

    poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]
    poisson_norm = (
        (baseline["baseline_poisson"] - poisson) / poisson_range
        if poisson_range > 0
        else 0.0
    )

    if poisson_norm < 0.97:
        return False

    return True


class LocalDenoisingState:
    """
    Local replacement for examples.denoising.env.DenoisingState.

    We only keep the fields used by local_ttt_denoising_tiny.py and PUCT.
    """

    def __init__(
        self,
        timestep: int,
        construction: list[Any],
        code: str,
        value: Optional[float] = None,
        mse: Optional[float] = None,
        poisson: Optional[float] = None,
        parent_values: Optional[list[float]] = None,
        parents: Optional[list[dict]] = None,
        id: Optional[str] = None,
        observation: str = "",
    ):
        self.id = id or str(uuid.uuid4())
        self.timestep = timestep
        self.construction = construction
        self.code = code
        self.value = value
        self.mse = mse
        self.poisson = poisson
        self.parent_values = parent_values or []
        self.parents = parents or []
        self.observation = observation

    def to_dict(self) -> dict:
        return {
            "type": "LocalDenoisingState",
            "id": self.id,
            "timestep": self.timestep,
            "construction": self.construction,
            "code": self.code,
            "value": self.value,
            "mse": self.mse,
            "poisson": self.poisson,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data.get("id"),
            timestep=data["timestep"],
            construction=data.get("construction", []),
            code=data.get("code", ""),
            value=data.get("value"),
            mse=data.get("mse"),
            poisson=data.get("poisson"),
            parent_values=data.get("parent_values", []),
            parents=data.get("parents", []),
            observation=data.get("observation", ""),
        )


class LocalPUCTSampler:
    """
    Local PUCT sampler.

    This mirrors the important behaviour of the official PUCTSampler:

        score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n[i])

    where:
    - Q(i) is the best child value reachable from this state, or its own value
    - P(i) is a rank-based prior
    - n[i] is how often this state / its lineage has been visited
    - T is the total number of expansions
    """

    def __init__(
        self,
        file_path: str,
        env_type: type,
        problem_type: str = "",
        max_buffer_size: int = 1000,
        batch_size: int = 1,
        puct_c: float = 1.0,
        topk_children: int = 2,
    ):
        self.file_path = Path(file_path)
        self.env_type = env_type
        self.problem_type = problem_type
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.puct_c = float(puct_c)
        self.topk_children = topk_children

        self._states = []
        self._initial_states = []
        self._last_sampled_states = []
        self._last_sampled_indices = []
        self._last_scale = 1.0
        self._last_puct_stats = []

        self._n = {}
        self._m = {}
        self._T = 0
        self._current_step = 0

        self._load_if_exists()

        if not self._states:
            for _ in range(batch_size):
                state = self.env_type.create_initial_state(problem_type)
                self._initial_states.append(state)
                self._states.append(state)
            self._save(step=0)

    def _sampler_file_for_step(self, step: int) -> Path:
        base = str(self.file_path)
        if base.endswith(".json"):
            base = base[:-5]
        return Path(f"{base}_step_{step:06d}.json")

    def _latest_file(self):
        parent = self.file_path.parent
        stem = self.file_path.stem
        files = sorted(parent.glob(f"{stem}_step_*.json"))
        return files[-1] if files else None

    def _load_if_exists(self):
        latest = self._latest_file()
        if latest is None:
            return

        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            self._states = [
                self.env_type.state_type.from_dict(item)
                for item in data.get("states", [])
            ]
            self._initial_states = [
                self.env_type.state_type.from_dict(item)
                for item in data.get("initial_states", [])
            ]
            self._n = data.get("puct_n", {}) or {}
            self._m = data.get("puct_m", {}) or {}
            self._T = int(data.get("puct_T", 0) or 0)
            self._current_step = int(data.get("step", 0) or 0)
            print(f"Loaded local PUCT sampler from {latest}")
        except Exception as error:
            print("Warning: failed to load previous local PUCT sampler.")
            print("Starting from a new sampler.")
            print("Error:", repr(error))

    def _save(self, step: int):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        path = self._sampler_file_for_step(step)

        data = {
            "step": step,
            "states": [state.to_dict() for state in self._states],
            "initial_states": [state.to_dict() for state in self._initial_states],
            "puct_n": self._n,
            "puct_m": self._m,
            "puct_T": self._T,
        }

        tmp_path = Path(str(path) + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _get_key(self, state):
        if getattr(state, "code", None):
            return state.code
        if getattr(state, "construction", None):
            return tuple(state.construction)
        return state.id

    def _compute_scale(self, values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 1.0
        return float(max(np.max(finite) - np.min(finite), 1e-6))

    def _compute_prior(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.array([])

        safe_values = np.where(np.isfinite(values), values, -1e30)
        n = len(safe_values)

        ranks = np.argsort(np.argsort(-safe_values))
        weights = (n - ranks).astype(np.float64)
        weights_sum = weights.sum()

        if weights_sum <= 0:
            return np.ones(n, dtype=np.float64) / n

        return weights / weights_sum

    def _set_parent_info(self, child, parent):
        child.parent_values = (
            [parent.value] + parent.parent_values
            if parent.value is not None
            else list(parent.parent_values)
        )
        child.parents = (
            [{"id": parent.id, "timestep": parent.timestep}]
            + list(parent.parents)
        )

    def sample_states(self, num_states: int):
        candidates = list(self._states)

        if not candidates:
            picked = [
                self.env_type.create_initial_state(self.problem_type)
                for _ in range(num_states)
            ]
            self._last_sampled_states = picked
            self._last_sampled_indices = []
            return picked

        values = np.array(
            [
                float(state.value)
                if state.value is not None
                else float("-inf")
                for state in candidates
            ],
            dtype=np.float64,
        )

        scale = self._compute_scale(values)
        self._last_scale = scale

        prior = self._compute_prior(values)
        sqrt_t = math.sqrt(1.0 + self._T)

        scored = []

        for i, state in enumerate(candidates):
            n_i = self._n.get(state.id, 0)
            m_i = self._m.get(state.id, values[i])
            q_i = m_i if n_i > 0 else values[i]

            bonus = self.puct_c * scale * prior[i] * sqrt_t / (1.0 + n_i)
            score = q_i + bonus

            scored.append((score, values[i], i, state, n_i, q_i, prior[i], bonus))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected = scored[:num_states]
        picked = [item[3] for item in selected]

        self._last_sampled_states = picked
        self._last_sampled_indices = [item[2] for item in selected]
        self._last_puct_stats = [
            {
                "n": item[4],
                "Q": float(item[5]),
                "P": float(item[6]),
                "bonus": float(item[7]),
                "score": float(item[0]),
            }
            for item in selected
        ]

        return picked

    def update_states(
        self,
        states: list,
        parent_states: list,
        save: bool = True,
        step: Optional[int] = None,
    ):
        if not states:
            return

        if len(states) != len(parent_states):
            raise ValueError("states and parent_states must have same length")

        parent_best_child = {}

        for child, parent in zip(states, parent_states):
            if child.value is None:
                continue

            child_value = float(child.value)
            parent_best_child[parent.id] = max(
                parent_best_child.get(parent.id, float("-inf")),
                child_value,
            )

        for child, parent in zip(states, parent_states):
            parent_id = parent.id

            if parent_id in parent_best_child:
                best_value = parent_best_child[parent_id]
                self._m[parent_id] = max(
                    self._m.get(parent_id, best_value),
                    best_value,
                )

            ancestor_ids = [parent.id] + [
                str(p["id"])
                for p in parent.parents
                if p.get("id")
            ]

            for ancestor_id in ancestor_ids:
                self._n[ancestor_id] = self._n.get(ancestor_id, 0) + 1

            self._T += 1

        # Keep only top-k children per parent.
        if self.topk_children and self.topk_children > 0:
            grouped = {}

            for child, parent in zip(states, parent_states):
                grouped.setdefault(parent.id, []).append((child, parent))

            filtered_states = []
            filtered_parents = []

            for pairs in grouped.values():
                pairs.sort(
                    key=lambda pair: (
                        pair[0].value
                        if pair[0].value is not None
                        else float("-inf")
                    ),
                    reverse=True,
                )

                for child, parent in pairs[: self.topk_children]:
                    filtered_states.append(child)
                    filtered_parents.append(parent)

            states = filtered_states
            parent_states = filtered_parents

        existing_keys = {self._get_key(state) for state in self._states}

        new_states = []

        for child, parent in zip(states, parent_states):
            if child.value is None:
                continue

            key = self._get_key(child)

            if key in existing_keys:
                continue

            self._set_parent_info(child, parent)
            new_states.append(child)
            existing_keys.add(key)

        self._states.extend(new_states)

        if len(self._states) > self.max_buffer_size:
            self._states.sort(
                key=lambda state: (
                    state.value
                    if state.value is not None
                    else float("-inf")
                ),
                reverse=True,
            )
            self._states = self._states[: self.max_buffer_size]

        if save:
            self.flush(step=step)

    def record_failed_rollout(self, parent):
        ancestor_ids = [parent.id] + [
            str(p["id"])
            for p in parent.parents
            if p.get("id")
        ]

        for ancestor_id in ancestor_ids:
            self._n[ancestor_id] = self._n.get(ancestor_id, 0) + 1

        self._T += 1

    def flush(self, step: Optional[int] = None):
        if step is not None:
            self._current_step = step

        self._save(step=self._current_step)

    def get_sample_stats(self) -> dict:
        values = [
            state.value
            for state in self._states
            if state.value is not None
        ]

        sampled_values = [
            state.value
            for state in self._last_sampled_states
            if state.value is not None
        ]

        def stats(prefix, xs):
            if not xs:
                return {}

            arr = np.array(xs, dtype=np.float64)

            return {
                f"{prefix}/mean": float(arr.mean()),
                f"{prefix}/std": float(arr.std()),
                f"{prefix}/min": float(arr.min()),
                f"{prefix}/max": float(arr.max()),
            }

        out = {
            "puct/buffer_size": len(self._states),
            "puct/sampled_size": len(self._last_sampled_states),
            "puct/T": self._T,
            "puct/scale_last": float(self._last_scale),
        }

        out.update(stats("puct/buffer_value", values))
        out.update(stats("puct/sampled_value", sampled_values))

        return out
