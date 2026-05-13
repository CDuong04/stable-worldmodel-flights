"""Action-space adapters for model-based planners.

The world models are trained on normalized, frameskip-packed actions, while
gym environments execute primitive physical actions. These helpers make that
interface explicit so CEM scores and executes the same bounded action sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk


@dataclass
class RocketActionNormalizer:
    """Normalize/invert primitive or frameskip-packed rocket actions."""

    mean: np.ndarray
    std: np.ndarray
    action_block: int = 1
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(self.std, dtype=np.float32), self.eps)
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("RocketActionNormalizer mean/std must be 1-D")
        if self.mean.shape != self.std.shape:
            raise ValueError("RocketActionNormalizer mean/std shapes must match")
        self.primitive_dim = int(self.mean.shape[0])

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        action_block: int = 1,
        eps: float = 1e-6,
    ) -> "RocketActionNormalizer":
        ds = load_from_disk(str(dataset_path))
        actions = np.asarray(ds["action"], dtype=np.float32)
        return cls(
            mean=actions.mean(axis=0),
            std=np.maximum(actions.std(axis=0), eps),
            action_block=action_block,
            eps=eps,
        )

    def _stats_for(self, x):
        last_dim = int(x.shape[-1])
        if last_dim == self.primitive_dim:
            mean = self.mean
            std = self.std
        elif last_dim == self.primitive_dim * self.action_block:
            mean = np.tile(self.mean, self.action_block)
            std = np.tile(self.std, self.action_block)
        else:
            raise ValueError(
                f"Expected last action dim {self.primitive_dim} or "
                f"{self.primitive_dim * self.action_block}, got {last_dim}"
            )

        if torch.is_tensor(x):
            mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(std, device=x.device, dtype=x.dtype)
        else:
            mean = mean.astype(np.asarray(x).dtype, copy=False)
            std = std.astype(np.asarray(x).dtype, copy=False)
        return mean, std

    def transform(self, x):
        mean, std = self._stats_for(x)
        return (x - mean) / std

    def inverse_transform(self, x):
        mean, std = self._stats_for(x)
        return x * std + mean


@dataclass
class DatasetColumnNormalizer:
    """Normalize/invert a fixed-width dataset column."""

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(self.std, dtype=np.float32), self.eps)
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("DatasetColumnNormalizer mean/std must be 1-D")
        if self.mean.shape != self.std.shape:
            raise ValueError("DatasetColumnNormalizer mean/std shapes must match")

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        column: str,
        eps: float = 1e-6,
    ) -> "DatasetColumnNormalizer":
        ds = load_from_disk(str(dataset_path))
        values = np.asarray(ds[column], dtype=np.float32)
        return cls(
            mean=values.mean(axis=0),
            std=np.maximum(values.std(axis=0), eps),
            eps=eps,
        )

    def _stats_for(self, x):
        last_dim = int(x.shape[-1])
        if last_dim != int(self.mean.shape[0]):
            raise ValueError(
                f"Expected last dim {self.mean.shape[0]}, got {last_dim}"
            )

        if torch.is_tensor(x):
            mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        else:
            arr = np.asarray(x)
            mean = self.mean.astype(arr.dtype, copy=False)
            std = self.std.astype(arr.dtype, copy=False)
        return mean, std

    def transform(self, x):
        mean, std = self._stats_for(x)
        return (x - mean) / std

    def inverse_transform(self, x):
        mean, std = self._stats_for(x)
        return x * std + mean


@dataclass
class RocketActionAdapter:
    """Project normalized rocket candidates through physical action bounds."""

    normalizer: RocketActionNormalizer
    low: tuple[float, ...] = (-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0)
    high: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    action_block: int = 1

    def __post_init__(self) -> None:
        self.low_arr = np.asarray(self.low, dtype=np.float32)
        self.high_arr = np.asarray(self.high, dtype=np.float32)
        if self.low_arr.shape != (self.normalizer.primitive_dim,):
            raise ValueError("RocketActionAdapter bounds must match primitive action dim")
        if self.action_block != self.normalizer.action_block:
            self.action_block = self.normalizer.action_block

    def _bounds_for(self, x):
        if torch.is_tensor(x):
            low = torch.as_tensor(self.low_arr, device=x.device, dtype=x.dtype)
            high = torch.as_tensor(self.high_arr, device=x.device, dtype=x.dtype)
        else:
            arr = np.asarray(x)
            low = self.low_arr.astype(arr.dtype, copy=False)
            high = self.high_arr.astype(arr.dtype, copy=False)
        return low, high

    def project_raw(self, raw_actions):
        """Clamp primitive or packed raw actions to the rocket env bounds."""
        primitive_dim = self.normalizer.primitive_dim
        original_shape = raw_actions.shape
        if int(original_shape[-1]) == primitive_dim:
            flat = raw_actions.reshape(-1, primitive_dim)
        elif int(original_shape[-1]) == primitive_dim * self.action_block:
            flat = raw_actions.reshape(-1, self.action_block, primitive_dim)
        else:
            raise ValueError(
                f"Expected last action dim {primitive_dim} or "
                f"{primitive_dim * self.action_block}, got {original_shape[-1]}"
            )

        low, high = self._bounds_for(raw_actions)
        if torch.is_tensor(raw_actions):
            projected = torch.clamp(flat, min=low, max=high)
        else:
            projected = np.clip(flat, low, high)
        return projected.reshape(original_shape)

    def project_normalized(self, normalized_actions):
        """Project normalized actions by inverting, clamping, and renormalizing."""
        raw = self.normalizer.inverse_transform(normalized_actions)
        projected_raw = self.project_raw(raw)
        return self.normalizer.transform(projected_raw)


class ActionSpaceCostAdapter:
    """Wrap a cost model so all scored candidates are action-space projected."""

    def __init__(self, base_cost_model, action_adapter: RocketActionAdapter):
        self.base = base_cost_model
        self.action_adapter = action_adapter

    def project_actions(self, action_candidates: torch.Tensor) -> torch.Tensor:
        return self.action_adapter.project_normalized(action_candidates)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        return self.base.get_cost(info_dict, self.project_actions(action_candidates))

    def diagnose_plan(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        if hasattr(self.base, "diagnose_plan"):
            return self.base.diagnose_plan(info_dict, self.project_actions(action_candidates))
        return {"available": False, "reason": "wrapped cost has no diagnose_plan"}

    def select_shielded_actions(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
        costs: torch.Tensor,
    ):
        if hasattr(self.base, "select_shielded_actions"):
            projected = self.project_actions(action_candidates)
            actions, diag = self.base.select_shielded_actions(info_dict, projected, costs)
            if isinstance(diag, dict):
                diag["actions_projected"] = True
            return actions, diag
        best = torch.argmin(costs, dim=1)
        batch = torch.arange(action_candidates.shape[0], device=action_candidates.device)
        return action_candidates[batch, best], {"shield_active": False}

    def eval(self):
        self.base.eval()
        return self

    def to(self, device):
        self.base.to(device)
        return self

    def __getattr__(self, name: str):
        return getattr(self.base, name)
