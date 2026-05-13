"""Smoke checks for the RocketJEPA action-space adapter."""
from __future__ import annotations

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_worldmodel.solver.action_adapter import (
    ActionSpaceCostAdapter,
    RocketActionAdapter,
    RocketActionNormalizer,
)


class _DummyCost:
    def get_cost(self, info_dict, action_candidates):
        info_dict["seen_actions"] = action_candidates
        return action_candidates.square().sum(dim=(-1, -2))

    def eval(self):
        return self

    def to(self, device):
        return self


def main() -> None:
    mean = np.array([0.0, 0.0, 0.0, 0.9, 0.5, 0.0, 0.0], dtype=np.float32)
    std = np.array([0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2], dtype=np.float32)
    normalizer = RocketActionNormalizer(mean=mean, std=std, action_block=2)

    raw = np.array([[0.1, -0.2, 0.0, 1.0, 0.7, 0.3, -0.4]], dtype=np.float32)
    restored = normalizer.inverse_transform(normalizer.transform(raw))
    assert np.allclose(restored, raw)

    packed = np.concatenate([raw, raw], axis=-1)
    restored_packed = normalizer.inverse_transform(normalizer.transform(packed))
    assert np.allclose(restored_packed, packed)

    adapter = RocketActionAdapter(normalizer=normalizer, action_block=2)
    raw_bad = np.array([[2.0, -2.0, 0.0, -1.0, 3.0, -4.0, 4.0]], dtype=np.float32)
    raw_projected = adapter.project_raw(raw_bad)
    assert np.all(raw_projected >= np.array(adapter.low) - 1e-6)
    assert np.all(raw_projected <= np.array(adapter.high) + 1e-6)

    norm_bad = torch.as_tensor(normalizer.transform(np.concatenate([raw_bad, raw_bad], axis=-1)))
    projected_norm = adapter.project_normalized(norm_bad)
    projected_raw = normalizer.inverse_transform(projected_norm.numpy())
    assert np.all(projected_raw[..., 0:7] >= np.array(adapter.low) - 1e-6)
    assert np.all(projected_raw[..., 0:7] <= np.array(adapter.high) + 1e-6)

    wrapped = ActionSpaceCostAdapter(_DummyCost(), adapter)
    candidates = norm_bad.reshape(1, 1, 1, 14)
    info = {}
    cost = wrapped.get_cost(info, candidates)
    assert cost.shape == (1, 1)
    assert torch.isfinite(cost).all()
    assert "seen_actions" in info

    print("rocket action adapter smoke checks passed")


if __name__ == "__main__":
    main()
