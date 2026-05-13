"""Pure-physics MPC cost — plug into CEMSolver in place of a world model."""

from __future__ import annotations

import torch

from stable_worldmodel.solver.rocket_physics import (
    RocketConstraintCost,
    RocketPhysicsModel,
    project_rocket_actions,
)


class PhysicsOnlyCost:
    """Score action candidates purely via rocket physics + soft constraint penalties.

    Mimics the subset of the world-model API that CEMSolver and WorldModelPolicy
    actually call: `.get_cost(info_dict, action_candidates)`, plus a handful of
    no-op methods (`.to`, `.eval`, `.requires_grad_`) so the caller can treat
    this object like any torch module.
    """

    def __init__(
        self,
        physics_model: RocketPhysicsModel | None = None,
        constraint_cost: RocketConstraintCost | None = None,
        project_actions: bool = True,
        device: str = "cuda",
    ) -> None:
        self.physics = (physics_model or RocketPhysicsModel()).to(device)
        self.constraints = (constraint_cost or RocketConstraintCost()).to(device)
        self.project = project_actions
        self.device = device
        # Attr some callers may poke; harmless here.
        self.interpolate_pos_encoding = True

    def to(self, *args, **kwargs):
        self.physics = self.physics.to(*args, **kwargs)
        self.constraints = self.constraints.to(*args, **kwargs)
        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                self.device = str(first)
        return self

    def eval(self):
        return self

    def requires_grad_(self, *args, **kwargs):
        return self

    def parameters(self):
        return iter([])

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """CEM passes candidates shaped (B, N, H, D); rollout wants (M, H, D).

        Flattens, rolls out, reshapes costs back to (B, N).
        """
        if self.project:
            action_candidates = project_rocket_actions(action_candidates)

        actions = action_candidates.to(self.device).float()
        B, N, H, D = actions.shape

        proprio = info_dict["proprio"]
        if torch.is_tensor(proprio):
            state_0 = proprio.to(self.device).float()
        else:
            state_0 = torch.from_numpy(proprio).to(self.device).float()

        # CEM pre-expands info to (B, N, ...). For proprio that's (B, N, history, 17).
        # Take the latest history frame → (B, N, 17) → flatten to (B*N, 17).
        if state_0.ndim == 4:
            state_0 = state_0[:, :, -1]
        elif state_0.ndim == 3:
            # (B, history, 17) — not yet sample-expanded. Expand.
            state_0 = state_0[:, -1].unsqueeze(1).expand(B, N, -1)
        elif state_0.ndim == 2:
            # (B, 17)
            state_0 = state_0.unsqueeze(1).expand(B, N, -1)
        state_rep = state_0.reshape(B * N, -1)
        actions_flat = actions.reshape(B * N, H, D)

        trajectory = self.physics.rollout(state_rep, actions_flat)  # (B*N, H+1, 17)
        costs_flat = self.constraints(trajectory)  # (B*N,)
        return costs_flat.view(B, N)
