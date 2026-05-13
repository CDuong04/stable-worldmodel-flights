"""CLF-augmented CEM cost (rocket_jepa.tex eq:cem_clf_cost).

Wraps a base world-model cost with the Lyapunov-descent penalty along the
predicted latent rollout. The wrapped cost is what the paper's central
hypothesis (P1: standard MPC violates descent; P2: CLF-MPC preserves it)
tests.

Per [rocket_jepa.tex eq:cem_clf_cost]:

    C_CLF^(i) = ||z_H^(i) - z_goal||^2
              + lambda_V * sum_{k=0..H-1} max(0, V(x_hat_{t+k+1}^(i))
                                              - V(x_hat_{t+k}^(i)) + eta)

We do NOT touch dinowm.get_cost; instead, this wrapper exposes the same
get_cost(info_dict, action_candidates) interface and is plugged into
swm.solver.CEMSolver as the cost model.

Setting lambda_V=0 reproduces the base cost exactly, so the same script
can run the standard-CEM (P1) and CLF-MPC (P2) arms by just toggling the
weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class CLFCostConfig:
    """Hyperparameters for the CLF-augmented CEM cost.

    lambda_V: weight on the descent penalty. 0 disables CLF (standard CEM).
    eta: slack on the descent inequality. Small positive values require
        strict descent below -eta to avoid the penalty.
    hard_filter: if true, any candidate with a predicted descent violation
        receives a large lexicographic-style infeasibility penalty. This makes
        CEM prefer feasible CLF rollouts when they exist and fall back to the
        least-violating rollout otherwise.
    """
    lambda_V: float = 0.0
    eta: float = 0.0
    normalized_descent: bool = True
    eps: float = 1e-6
    hard_filter: bool = False
    hard_filter_weight: float = 1e6
    runtime_shield: bool = False
    shield_margin: float = 0.0
    calibration_epsilon: float = 0.0
    shield_full_rollout: bool = True
    shield_use_v_head: bool = False
    shield_use_delta_head: bool = False
    use_v_head: bool = True
    use_delta_head: bool = True


class CLFAugmentedCost:
    """Augment any world-model cost with a decoded-V CLF descent penalty.

    Args:
        base_cost_model: any module with `.get_cost(info_dict, action_candidates) -> Tensor`
            and an `.encode/.rollout` API matching dinowm. The wrapper will
            invoke `.rollout` itself to get the predicted latent trajectory
            and a separate forward pass for the original goal cost.
        decoder: state probe d_phi: (B, embed_dim) -> (B, state_dim).
            Already trained; should be in eval mode.
        compute_V: function (B, state_dim) -> (B,) returning the Lyapunov
            value V(x_hat) for the rocket task. Must operate on tensors
            with autograd disabled (CEM uses inference_mode).
        cfg: CLFCostConfig (lambda_V, eta).
        device: where to run the augmentation.

    The wrapper is itself a Costable -- it owns no parameters of its own.
    """

    def __init__(
        self,
        base_cost_model,
        decoder: torch.nn.Module,
        compute_V: Callable[[torch.Tensor], torch.Tensor],
        cfg: Optional[CLFCostConfig] = None,
        device: str | torch.device = "cpu",
    ):
        self.base = base_cost_model
        self.decoder = decoder.to(device).eval()
        self.compute_V = compute_V
        self.cfg = cfg or CLFCostConfig()
        self.device = torch.device(device)

        # Mirror the base cost model's torch attributes so AutoCostModel +
        # CEMSolver introspection still works (n_history, etc).
        for attr in ("backbone", "world_model", "encode", "rollout",
                     "split_embedding"):
            if hasattr(base_cost_model, attr) and not hasattr(self, attr):
                setattr(self, attr, getattr(base_cost_model, attr))

    # -- internal helpers -----------------------------------------------------

    def _pooled_latents(self, predicted_latents: torch.Tensor) -> torch.Tensor:
        if predicted_latents.dim() == 4:
            return predicted_latents.mean(dim=2)
        if predicted_latents.dim() == 3:
            return predicted_latents
        raise ValueError(
            f"predicted_latents must be (B, T, P, d) or (B, T, d); "
            f"got {tuple(predicted_latents.shape)}"
        )

    def _v_head(self):
        return getattr(self.base, "v_head", None)

    def _delta_head(self):
        return getattr(self.base, "delta_head", None)

    def _decode_traj_and_V(
        self,
        predicted_latents: torch.Tensor,
        *,
        use_v_head: bool | None = None,
    ) -> torch.Tensor:
        """Decode each timestep of a predicted latent trajectory and compute V.

        predicted_latents: (B, T, P, d) -- output of dinowm.rollout's
            'predicted_pixels_embed'. We pool over patches (mean) and decode
            per timestep.

        Returns: (B, T) Lyapunov values along the rollout.
        """
        z = self._pooled_latents(predicted_latents)
        if use_v_head is None:
            use_v_head = self.cfg.use_v_head
        v_head = self._v_head()
        if use_v_head and v_head is not None:
            B, T, d = z.shape
            logV = v_head(z.reshape(B * T, d)).reshape(B, T)
            target = getattr(self.base, "v_head_target", "log1p")
            if target == "log1p":
                return torch.expm1(logV).clamp_min(0.0)
            return logV.clamp_min(0.0)

        B, T, d = z.shape
        z_flat = z.reshape(B * T, d)
        x_hat = self.decoder(z_flat)                        # (B*T, state_dim)
        V = self.compute_V(x_hat)                           # (B*T,)
        return V.reshape(B, T)

    def _predict_deltas(
        self,
        predicted_latents: torch.Tensor,
        V_traj: torch.Tensor,
        *,
        use_delta_head: bool | None = None,
    ) -> torch.Tensor:
        if use_delta_head is None:
            use_delta_head = self.cfg.use_delta_head
        delta_head = self._delta_head()
        if use_delta_head and delta_head is not None:
            z = self._pooled_latents(predicted_latents)
            if z.shape[1] < 2:
                return torch.zeros(z.shape[0], 0, device=z.device, dtype=z.dtype)
            B, T, d = z.shape
            return delta_head(
                z[:, :-1].reshape(B * (T - 1), d),
                z[:, 1:].reshape(B * (T - 1), d),
            ).reshape(B, T - 1)
        return self._monitor_deltas(V_traj)

    def _clf_penalty(self, V_traj: torch.Tensor, deltas: torch.Tensor | None = None) -> torch.Tensor:
        """Compute sum_k max(0, descent_violation + eta) along the rollout.

        V_traj: (B, T)  ->  (B,) penalty scalar per candidate.
        """
        step_violation = self._clf_step_violation(V_traj, deltas=deltas)
        if step_violation.shape[1] < 1:
            return torch.zeros(V_traj.shape[0], device=V_traj.device,
                               dtype=V_traj.dtype)
        return step_violation.sum(dim=1)                    # (B,)

    def _clf_step_violation(
        self,
        V_traj: torch.Tensor,
        deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-step positive part of the CLF descent inequality."""
        if deltas is None:
            deltas = self._monitor_deltas(V_traj)
        return F.relu(self._raw_step_violation_from_deltas(deltas))

    def _raw_step_violation_from_deltas(self, deltas: torch.Tensor) -> torch.Tensor:
        if deltas.shape[1] < 1:
            return torch.zeros(deltas.shape[0], 0, device=deltas.device,
                               dtype=deltas.dtype)
        # deltas use the LyapunovMonitor convention: positive means descent.
        return -deltas + self.cfg.eta + float(self.cfg.calibration_epsilon)

    def _hard_filter_penalty(
        self,
        V_traj: torch.Tensor,
        deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Large infeasibility cost for candidates that violate CLF descent."""
        step_violation = self._clf_step_violation(V_traj, deltas=deltas)
        if step_violation.shape[1] < 1:
            return torch.zeros(V_traj.shape[0], device=V_traj.device,
                               dtype=V_traj.dtype)
        max_violation = step_violation.max(dim=1).values
        infeasible = (max_violation > 0).to(dtype=V_traj.dtype)
        return self.cfg.hard_filter_weight * (infeasible + max_violation)

    def _monitor_deltas(self, V_traj: torch.Tensor) -> torch.Tensor:
        """Return LyapunovMonitor-style deltas along a predicted rollout.

        Positive values mean V decreases. Negative values are descent
        violations. The CLF cost penalizes the negative of this quantity.
        """
        if V_traj.shape[1] < 2:
            return torch.zeros(V_traj.shape[0], 0, device=V_traj.device,
                               dtype=V_traj.dtype)
        if self.cfg.normalized_descent:
            return (V_traj[:, :-1] - V_traj[:, 1:]) / (
                V_traj[:, :-1].abs() + self.cfg.eps
            )
        return V_traj[:, :-1] - V_traj[:, 1:]

    def _first_exec_idx(self, info_dict: dict, n_deltas: int) -> int:
        n_obs = 1
        pixels = info_dict.get("pixels")
        if torch.is_tensor(pixels) and pixels.dim() >= 2:
            n_obs = int(pixels.shape[1])
        return min(max(n_obs - 1, 0), max(n_deltas - 1, 0))

    def diagnose_plan(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        """Score a selected plan with the decoded CLF monitor.

        This is intentionally separate from ``get_cost`` so evaluation scripts
        can compare what the planner believed about the chosen rollout against
        the realized closed-loop monitor trace after stepping the simulator.
        """
        info = dict(info_dict)
        with torch.no_grad():
            base_cost = self.base.get_cost(info, action_candidates)
            if "predicted_pixels_embed" not in info:
                return {"available": False, "reason": "missing predicted_pixels_embed"}

            predicted = info["predicted_pixels_embed"]
            V_traj = self._decode_traj_and_V(predicted)
            deltas = self._predict_deltas(predicted, V_traj)
            raw_violation = self._raw_step_violation_from_deltas(deltas)
            step_violation = F.relu(raw_violation)
            penalty = self._clf_penalty(V_traj, deltas=deltas)
            hard_penalty = (
                self._hard_filter_penalty(V_traj, deltas=deltas)
                if self.cfg.hard_filter
                else torch.zeros_like(penalty)
            )

            base_flat = base_cost.reshape(-1).detach().cpu()
            penalty_flat = penalty.reshape(-1).detach().cpu()
            hard_flat = hard_penalty.reshape(-1).detach().cpu()
            total_flat = base_flat + float(self.cfg.lambda_V) * penalty_flat + hard_flat
            V_cpu = V_traj.detach().cpu()
            d_cpu = deltas.detach().cpu()
            step_v_cpu = step_violation.detach().cpu()

        if d_cpu.shape[1] > 0:
            violation_rate = (d_cpu < 0).float().mean(dim=1)
            mean_descent = d_cpu.mean(dim=1)
            min_descent = d_cpu.min(dim=1).values
            first_delta = d_cpu[:, 0]
            max_step_violation = step_v_cpu.max(dim=1).values
            feasible = max_step_violation <= 0
            exec_idx = self._first_exec_idx(info_dict, d_cpu.shape[1])
            exec_delta = d_cpu[:, exec_idx]
            exec_step_violation = step_v_cpu[:, exec_idx]
        else:
            n = V_cpu.shape[0]
            violation_rate = torch.zeros(n)
            mean_descent = torch.zeros(n)
            min_descent = torch.zeros(n)
            first_delta = torch.zeros(n)
            max_step_violation = torch.zeros(n)
            feasible = torch.ones(n, dtype=torch.bool)
            exec_idx = 0
            exec_delta = torch.zeros(n)
            exec_step_violation = torch.zeros(n)

        return {
            "available": True,
            "lambda_V": float(self.cfg.lambda_V),
            "eta": float(self.cfg.eta),
            "calibration_epsilon": float(self.cfg.calibration_epsilon),
            "normalized_descent": bool(self.cfg.normalized_descent),
            "using_v_head": bool(self.cfg.use_v_head and self._v_head() is not None),
            "using_delta_head": bool(self.cfg.use_delta_head and self._delta_head() is not None),
            "hard_filter": bool(self.cfg.hard_filter),
            "hard_filter_weight": float(self.cfg.hard_filter_weight),
            "base_cost": [float(x) for x in base_flat.tolist()],
            "clf_penalty": [float(x) for x in penalty_flat.tolist()],
            "hard_filter_penalty": [float(x) for x in hard_flat.tolist()],
            "total_cost": [float(x) for x in total_flat.tolist()],
            "pred_V_trace": [[float(x) for x in row] for row in V_cpu.tolist()],
            "pred_delta_trace": [[float(x) for x in row] for row in d_cpu.tolist()],
            "pred_step_violation_trace": [[float(x) for x in row] for row in step_v_cpu.tolist()],
            "pred_violation_rate": [float(x) for x in violation_rate.tolist()],
            "pred_mean_descent": [float(x) for x in mean_descent.tolist()],
            "pred_min_descent": [float(x) for x in min_descent.tolist()],
            "pred_first_delta": [float(x) for x in first_delta.tolist()],
            "pred_exec_idx": int(exec_idx),
            "pred_exec_delta": [float(x) for x in exec_delta.tolist()],
            "pred_exec_step_violation": [float(x) for x in exec_step_violation.tolist()],
            "pred_max_step_violation": [float(x) for x in max_step_violation.tolist()],
            "pred_feasible": [bool(x) for x in feasible.tolist()],
        }

    def select_shielded_actions(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
        costs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Pick executable plans with a calibrated CLF shield.

        The soft CLF objective can be drowned out by goal cost or CEM averaging.
        This selector is a final safety filter over the sampled candidate pool:
        prefer candidates whose predicted rollout satisfies the CLF descent
        inequality, breaking ties by the original CEM cost. If no candidate is
        feasible, choose the least-violating candidate and use cost only as a
        tiny tie-breaker.
        """
        if not self.cfg.runtime_shield or "predicted_pixels_embed" not in info_dict:
            best = torch.argmin(costs, dim=1)
            batch = torch.arange(action_candidates.shape[0], device=action_candidates.device)
            return action_candidates[batch, best], {"shield_active": False}

        with torch.no_grad():
            predicted = info_dict["predicted_pixels_embed"]
            V_traj = self._decode_traj_and_V(
                predicted,
                use_v_head=self.cfg.shield_use_v_head,
            )
            deltas = self._predict_deltas(
                predicted,
                V_traj,
                use_delta_head=self.cfg.shield_use_delta_head,
            )
            raw_violation = self._raw_step_violation_from_deltas(deltas)
            step_violation = F.relu(raw_violation)
            if raw_violation.shape[1] == 0:
                best = torch.argmin(costs, dim=1)
                batch = torch.arange(action_candidates.shape[0], device=action_candidates.device)
                return action_candidates[batch, best], {"shield_active": True, "reason": "empty_trace"}

            first_exec_idx = self._first_exec_idx(info_dict, raw_violation.shape[1])
            first_raw_violation = raw_violation[:, first_exec_idx].reshape(costs.shape)
            first_violation = step_violation[:, first_exec_idx].reshape(costs.shape)
            if self.cfg.shield_full_rollout:
                rollout_raw_violation = raw_violation[:, first_exec_idx:]
                rollout_step_violation = step_violation[:, first_exec_idx:]
                max_raw_violation = rollout_raw_violation.max(dim=1).values.reshape(costs.shape)
                max_step_violation = rollout_step_violation.max(dim=1).values.reshape(costs.shape)
                sum_step_violation = rollout_step_violation.sum(dim=1).reshape(costs.shape)
                feasible = max_raw_violation <= float(self.cfg.shield_margin)
                shield_violation = max_raw_violation
            else:
                max_raw_violation = first_raw_violation
                max_step_violation = first_violation
                sum_step_violation = first_violation
                feasible = first_raw_violation <= float(self.cfg.shield_margin)
                shield_violation = first_raw_violation

            # Lexicographic score: feasible candidates are sorted by original
            # cost; if none are feasible, sort primarily by violation.
            finite_cost = torch.nan_to_num(costs, nan=1e9, posinf=1e9, neginf=-1e9)
            cost_scale = finite_cost.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
            feasible_score = torch.where(
                feasible,
                finite_cost,
                torch.full_like(finite_cost, 1e12),
            )
            fallback_score = (
                shield_violation
                + 1e-3 * sum_step_violation
                + 1e-6 * finite_cost / cost_scale
            )
            any_feasible = feasible.any(dim=1, keepdim=True)
            score = torch.where(any_feasible, feasible_score, fallback_score)
            best = torch.argmin(score, dim=1)
            batch = torch.arange(action_candidates.shape[0], device=action_candidates.device)
            chosen_violation = first_violation[batch, best]
            chosen_raw_violation = first_raw_violation[batch, best]
            chosen_max_violation = max_step_violation[batch, best]
            chosen_max_raw_violation = max_raw_violation[batch, best]
            chosen_sum_violation = sum_step_violation[batch, best]

        return action_candidates[batch, best], {
            "shield_active": True,
            "shield_full_rollout": bool(self.cfg.shield_full_rollout),
            "shield_first_exec_idx": int(first_exec_idx),
            "shield_calibration_epsilon": float(self.cfg.calibration_epsilon),
            "shield_using_v_head": bool(self.cfg.shield_use_v_head and self._v_head() is not None),
            "shield_using_delta_head": bool(self.cfg.shield_use_delta_head and self._delta_head() is not None),
            "shield_feasible_rate": [float(x) for x in feasible.float().mean(dim=1).detach().cpu().tolist()],
            "shield_any_feasible": [bool(x) for x in feasible.any(dim=1).detach().cpu().tolist()],
            "shield_chosen_violation": [float(x) for x in chosen_violation.detach().cpu().tolist()],
            "shield_chosen_raw_violation": [float(x) for x in chosen_raw_violation.detach().cpu().tolist()],
            "shield_chosen_max_violation": [float(x) for x in chosen_max_violation.detach().cpu().tolist()],
            "shield_chosen_max_raw_violation": [float(x) for x in chosen_max_raw_violation.detach().cpu().tolist()],
            "shield_chosen_sum_violation": [float(x) for x in chosen_sum_violation.detach().cpu().tolist()],
        }

    # -- Costable interface ---------------------------------------------------

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Augmented cost: base goal cost + lambda_V * CLF descent penalty.

        Returns the same shape as base.get_cost (CEM expects it).
        """
        # 1. Base cost (also runs encode + rollout and populates info_dict).
        base_cost = self.base.get_cost(info_dict, action_candidates)

        # If CLF is disabled, return base cost unchanged. This makes lambda_V=0
        # byte-identical to standard CEM, so the same script can run both arms.
        if self.cfg.lambda_V == 0.0:
            return base_cost

        # 2. Pull the predicted latent trajectory that dinowm just stashed in
        # info_dict during the rollout step inside base.get_cost.
        if "predicted_pixels_embed" not in info_dict:
            # Defensive: fall back to base cost if the cache isn't there
            # (e.g., a non-dinowm cost model).
            return base_cost
        predicted = info_dict["predicted_pixels_embed"]      # (B, T, P, d)

        # 3. Decode + compute V along the rollout.
        with torch.no_grad():
            V_traj = self._decode_traj_and_V(predicted)
            deltas = self._predict_deltas(predicted, V_traj)
            penalty = self._clf_penalty(V_traj, deltas=deltas)  # (B,)
            hard_penalty = (
                self._hard_filter_penalty(V_traj, deltas=deltas)
                if self.cfg.hard_filter
                else torch.zeros_like(penalty)
            )

        # Match base_cost shape (it may be 1-D (B,) or 2-D (B, N))
        penalty = penalty.reshape(base_cost.shape)
        hard_penalty = hard_penalty.reshape(base_cost.shape)

        return base_cost + self.cfg.lambda_V * penalty + hard_penalty

    # CEMSolver introspects `.device`; some other code may look for `.eval()`
    def eval(self):
        self.base.eval()
        self.decoder.eval()
        return self

    def to(self, device):
        self.base.to(device)
        self.decoder.to(device)
        self.device = torch.device(device)
        return self


def make_rocket_V(alpha: float = 0.5, beta: float = 0.1,
                  pad_pos=(0.0, 0.0, 0.0),
                  pos_idx=(0, 1, 2), vel_idx=(3, 4, 5),
                  omega_idx=(10, 11, 12)) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the rocket-landing Lyapunov function as a callable for CLFAugmentedCost.

    V(x_hat) = ||p - p_pad||^2 + alpha * ||v||^2 + beta * ||omega||^2

    Same form as stable_worldmodel.lyapunov.LyapunovMonitor.compute_V but
    callable on a (B, state_dim) tensor.
    """
    pad = torch.as_tensor(pad_pos, dtype=torch.float32)

    def V(x_hat: torch.Tensor) -> torch.Tensor:
        p = x_hat[..., list(pos_idx)]
        v = x_hat[..., list(vel_idx)]
        omega = x_hat[..., list(omega_idx)]
        pad_dev = pad.to(device=x_hat.device, dtype=x_hat.dtype)
        return ((p - pad_dev) ** 2).sum(-1) + alpha * (v ** 2).sum(-1) + beta * (omega ** 2).sum(-1)

    return V
