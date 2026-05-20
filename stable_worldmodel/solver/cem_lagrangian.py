"""Hybrid CEM-Lagrangian solver.

Combines CEM's sample-based exploration with the augmented-Lagrangian
constraint machinery. For each outer dual-ascent iteration:
    1. Sample K candidates from N(mu, sigma).
    2. Score each by the augmented Lagrangian
           L = J + lambda^T g + 0.5 * rho * ||ReLU(g)||^2.
    3. Take top-M elites, refit (mu, sigma).
    4. Update multipliers via dual ascent on the elite mean's constraint values.
    5. Amplify rho.

The first action of the best elite is executed.

This addresses the limitation of the pure-gradient LagrangianSolver, which
underperforms CEM on regions where constraints are inactive (because Adam
with a small num_samples has no informational advantage over CEM there).
The hybrid recovers CEM's exploration when constraints are slack and gains
augmented-Lagrangian's principled constraint handling when they bind.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class CEMLagrangianSolver:
    """Hybrid CEM + augmented-Lagrangian solver.

    Args:
        model: Cost model implementing get_cost and get_constraints.
        batch_size: Number of environments processed in parallel.
        num_samples: Number of CEM candidates per inner iteration.
        topk: Number of elite samples to keep for distribution refit.
        n_inner_iters: CEM iterations per outer dual-ascent step.
        n_outer_iters: Number of dual-ascent (outer) iterations.
        var_scale: Initial variance scale for the action distribution.
        rho_init: Initial penalty coefficient.
        rho_max: Maximum penalty coefficient.
        rho_scale: Multiplicative growth factor for rho.
        device: Torch device.
        seed: RNG seed.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 8,
        topk: int = 3,
        n_inner_iters: int = 4,
        n_outer_iters: int = 2,
        var_scale: float = 0.3,
        rho_init: float = 1.0,
        rho_max: float = 1e4,
        rho_scale: float = 1.5,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        diagnostics_enabled: bool = False,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.topk = topk
        self.n_inner_iters = n_inner_iters
        self.n_outer_iters = n_outer_iters
        self.var_scale = var_scale
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.rho_scale = rho_scale
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.diagnostics_enabled = diagnostics_enabled

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        if not isinstance(action_space, Box):
            logging.warning(f"Non-Box action space: {type(action_space)}")

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def init_action_distrib(self, actions: torch.Tensor | None = None):
        var = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim])
        mean = torch.zeros([self.n_envs, 0, self.action_dim]) if actions is None else actions
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            new = torch.zeros([self.n_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new], dim=1).to(mean.device)
        return mean, var

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        """Solve via CEM-Lagrangian iteration."""
        start_time = time.time()
        outputs = {"costs": [], "constraint_violations": [], "lambdas": [], "rhos": []}

        # ---- Initialize ----
        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)
        total_envs = self.n_envs

        selected = mean.clone()

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            # Expand info_dict to (current_bs, num_samples, ...).
            expanded = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    v_batch = v_batch.unsqueeze(1).expand(
                        current_bs, self.num_samples, *v_batch.shape[1:]
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                expanded[k] = v_batch

            # Initialize multipliers and rho.
            # Multipliers are per-constraint, lazily sized on first call.
            lambdas = None
            rho = self.rho_init

            best_actions = batch_mean.clone()
            best_costs = torch.full((current_bs,), float("inf"), device=self.device)

            for outer_iter in range(self.n_outer_iters):
                for inner_iter in range(self.n_inner_iters):
                    # Sample K candidates: (B, K, H, D)
                    cand = torch.randn(
                        current_bs, self.num_samples, self.horizon, self.action_dim,
                        generator=self.torch_gen, device=self.device,
                    )
                    cand = cand * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                    # Force first sample to be current mean
                    cand[:, 0] = batch_mean

                    if hasattr(self.model, "project_actions"):
                        cand = self.model.project_actions(cand)

                    # Use FRESH shallow-copies for each call: the cost model
                    # mutates info_dict by adding `goal_embed` and other keys,
                    # so reusing the same dict for both get_cost and
                    # get_constraints triggers an `already in info_dict` assert.
                    info_for_cost = dict(expanded)
                    base_costs = self.model.get_cost(info_for_cost, cand)  # (B, K)
                    assert base_costs.shape == (current_bs, self.num_samples), \
                        f"expected ({current_bs}, {self.num_samples}), got {base_costs.shape}"

                    constraints = None
                    if hasattr(self.model, "get_constraints"):
                        info_for_constr = dict(expanded)
                        constraints = self.model.get_constraints(info_for_constr, cand)  # (B, K, C)
                        if lambdas is None:
                            C = constraints.shape[-1]
                            lambdas = torch.zeros(current_bs, C, device=self.device)

                    if constraints is not None and constraints.shape[-1] > 0:
                        lam_expanded = lambdas.unsqueeze(1)  # (B, 1, C)
                        viol = torch.relu(constraints)
                        aug_term = (
                            (lam_expanded * constraints).sum(dim=-1)
                            + 0.5 * rho * (viol ** 2).sum(dim=-1)
                        )
                    else:
                        aug_term = torch.zeros_like(base_costs)

                    total = base_costs + aug_term

                    # Elite selection
                    elite_vals, elite_idx = torch.topk(total, k=self.topk, dim=1, largest=False)
                    batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                    elite_cand = cand[batch_indices, elite_idx]

                    # Refit
                    batch_mean = elite_cand.mean(dim=1)
                    batch_var = elite_cand.std(dim=1).clamp_min(0.05)

                    # Track best-ever action
                    iter_best_idx = elite_idx[:, 0]
                    iter_best_cost = elite_vals[:, 0]
                    iter_best_action = cand[torch.arange(current_bs, device=self.device), iter_best_idx]
                    improved = iter_best_cost < best_costs
                    best_actions = torch.where(
                        improved.unsqueeze(-1).unsqueeze(-1),
                        iter_best_action, best_actions,
                    )
                    best_costs = torch.minimum(best_costs, iter_best_cost)

                # Dual ascent on elite mean's constraint values (one update per outer iter)
                if constraints is not None and constraints.shape[-1] > 0:
                    # Use the mean elite constraint values for the dual update
                    elite_constraints = constraints[batch_indices, elite_idx]  # (B, topk, C)
                    elite_mean_g = elite_constraints.mean(dim=1)  # (B, C)
                    lambdas = torch.clamp_min(lambdas + rho * elite_mean_g, 0.0)
                    rho = min(rho * self.rho_scale, self.rho_max)

                outputs["lambdas"].append(
                    (lambdas.detach().cpu().tolist() if lambdas is not None else [])
                )
                outputs["rhos"].append(float(rho))
                outputs["costs"].append(float(best_costs.mean().item()))

            if hasattr(self.model, "project_actions"):
                best_actions = self.model.project_actions(best_actions)

            selected[start_idx:end_idx] = best_actions
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var

        actions = selected
        if hasattr(self.model, "project_actions"):
            actions = self.model.project_actions(actions)
        outputs["actions"] = actions.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        if self.diagnostics_enabled and hasattr(self.model, "diagnose_plan"):
            try:
                outputs["diagnostics"] = self.model.diagnose_plan(info_dict.copy(), actions)
            except Exception as e:
                outputs["diagnostics_error"] = f"{type(e).__name__}: {e}"

        print(f"CEM-Lagrangian solve time: {time.time() - start_time:.4f} seconds")
        return outputs
