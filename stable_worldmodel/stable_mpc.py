"""Stable-manifold-constrained cost for LagrangianSolver-based MPC.

Replaces (and supplements) the post-hoc decoded-V monitor in clf_cost.py.  This
cost exposes BOTH a differentiable goal cost AND a stack of inequality
constraints that force the predicted latent rollout onto the local stable
manifold of f_θ at the touchdown equilibrium.

Constraint stack (returned by get_constraints, shape (B, S, C)):
  C1  ‖P^u v_t‖² ≤ ε_u²              for t = 0..H       (manifold structural)
  C2  V_L(v_H) ≤ α V_L(v_0)          terminal only      (latent Lyapunov, 1 constraint)
  C3  V(d_φ(z_{t+1})) ≤ V(d_φ(z_t)) − η  for t = 0..H-1 (decoded safety belt)

C1+C2 are load-bearing for the practical-stability proof; C3 is the existing
monitor's signal promoted from soft penalty to hard constraint.  Toggling
individual constraints via cfg.use_* enables the ablation rows in the paper.

Spectral artifact loading: expects a .pt file produced by
scripts/fit_local_linear_model.py with keys:
  z_star, a_star, P_s, P_u, P_lyap, V_s_basis, A_s_basis, eigvals, n_s, n_u,
  gap, spectral_radius_stable
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class StableMPCConfig:
    """Hyperparameters for the stable-manifold-constrained cost.

    Attributes:
        eps_u: bound on ‖P^u v_t‖² (unstable-subspace energy).  Setting larger
            relaxes C1; smaller forces the rollout closer to E^s.
        alpha: per-step Lyapunov decay rate.  α∈(0,1); smaller demands faster
            decay.  Conservative default 0.99 matches a slow burn.
        eta: descent slack for C3.  Larger demands strict descent of decoded V.
        normalized_descent: if True, C3 uses (V_t − V_{t+1}) / (V_t + eps)
            (Δ rate, matches existing LyapunovMonitor); else absolute Δ.
        eps: numerical safety in the normalized descent denominator.
        use_C1_unstable_bound: include the unstable-subspace constraint.
        use_C2_latent_lyap: include the latent Lyapunov-decrease constraint.
        use_C3_decoded_lyap: include the decoded-V descent constraint.
            Default False if C1+C2 are doing the work; turn on for the
            combined "monitor + manifold" mode.
    """
    eps_u: float = 1.0
    alpha: float = 0.99
    eta: float = 0.0
    normalized_descent: bool = True
    eps: float = 1e-6
    use_C1_unstable_bound: bool = True
    use_C2_latent_lyap: bool = True
    use_C3_decoded_lyap: bool = True
    # Weight on the latent V_L terminal cost relative to the base pixel-MSE.
    cost_weight: float = 0.1
    # ---- Adaptive gating ----
    # When True, multiply cost_weight by a sigmoid of the recent prediction
    # innovation (||z_observed - f_theta(z_{t-1}, a_{t-1})||).  Low innovation
    # (easy: model accurate) -> w_t ~= 0, behaves like pure CEM.  High innovation
    # (extreme: model fails under wind) -> w_t ~= 1, full V_L gating + C3 active.
    adaptive_weight: bool = False
    # EMA smoothing coefficient for the innovation: i_t = (1-beta)*i_{t-1} + beta*new
    innovation_ema_beta: float = 0.2
    # Sigmoid center: w_t = sigmoid((D_t - threshold) / scale)
    innovation_threshold: float = 8.0
    innovation_scale: float = 4.0
    # When True, also gates the C3 hard constraint by the same activation.
    # If False, C3 is always active when use_C3_decoded_lyap is True.
    gate_c3_by_innovation: bool = True


class StableManifoldConstrainedCost:
    """Cost + constraints for LagrangianSolver.

    Mirrors CLFAugmentedCost's attribute-mirroring (backbone, world_model,
    encode, rollout, split_embedding) so it slots into the same eval harness.

    NOTE: not a torch.nn.Module — buffers are managed manually to match the
    existing CLFAugmentedCost pattern.  ``to(device)`` propagates to base +
    decoder + cached spectral tensors.
    """

    def __init__(
        self,
        base_cost_model,
        decoder: torch.nn.Module,
        compute_V: Callable[[torch.Tensor], torch.Tensor],
        spectral_artifact_path: str | Path,
        cfg: Optional[StableMPCConfig] = None,
        device: str | torch.device = "cpu",
    ):
        self.base = base_cost_model
        self.decoder = decoder.to(device).eval()
        self.compute_V = compute_V
        self.cfg = cfg or StableMPCConfig()
        self.device = torch.device(device)

        # Mirror base attributes for harness introspection (n_history, etc.).
        for attr in (
            "backbone", "world_model", "encode", "rollout", "split_embedding",
        ):
            if hasattr(base_cost_model, attr) and not hasattr(self, attr):
                setattr(self, attr, getattr(base_cost_model, attr))

        # Load the local-linear-model artefact produced offline.
        artifact = torch.load(
            spectral_artifact_path,
            map_location=self.device,
            weights_only=False,
        )
        self._spectral_path = str(spectral_artifact_path)
        # Required keys; raise a clear error if the artefact was built differently.
        for key in ("z_star", "P_s", "P_u", "P_lyap"):
            if key not in artifact:
                raise KeyError(
                    f"spectral artefact missing required key '{key}': "
                    f"{spectral_artifact_path}"
                )
        self.z_star = artifact["z_star"].to(self.device).float()       # (d,)
        self.P_s = artifact["P_s"].to(self.device).float()             # (d, d)
        self.P_u = artifact["P_u"].to(self.device).float()             # (d, d)
        self.P_lyap = artifact["P_lyap"].to(self.device).float()       # (d, d)
        self.gap = float(artifact.get("gap", 0.0))
        self.spectral_radius_stable = float(
            artifact.get("spectral_radius_stable", 0.0)
        )
        self.n_s = int(artifact.get("n_s", -1))
        self.n_u = int(artifact.get("n_u", -1))

        # Single-use cache: LagrangianSolver calls get_cost then get_constraints
        # with the SAME action tensor; running the predictor twice would double
        # GPU memory.  Cache the predicted latents from get_cost so that
        # get_constraints can reuse them without a second rollout.  Keyed by
        # id() of the action tensor; cleared after one consumer reads it.
        self._predict_cache: tuple | None = None

        # Innovation EMA state for adaptive gating.  Updated externally per
        # control step (see runtime/eval harness).  Starts at threshold so the
        # very first plan uses w_t = 0.5; subsequent steps converge to the
        # actual disturbance level via EMA.
        self.innovation_ema: float = float(self.cfg.innovation_threshold)
        # Trace for diagnostics / paper figure.
        self.innovation_trace: list[float] = []
        self.activation_trace: list[float] = []

    # -- adaptive gating ------------------------------------------------------

    def update_innovation(self, innovation_norm: float) -> float:
        """Update the running EMA of the prediction innovation.

        Call once per control step from the eval harness with
        ||z_t_observed - f_theta(z_{t-1}, a_{t-1})||.  Returns the new EMA.
        """
        beta = float(self.cfg.innovation_ema_beta)
        self.innovation_ema = (1.0 - beta) * self.innovation_ema + beta * float(innovation_norm)
        self.innovation_trace.append(float(self.innovation_ema))
        self.activation_trace.append(float(self.activation_weight()))
        return self.innovation_ema

    def activation_weight(self) -> float:
        """sigmoid((innovation_ema - threshold) / scale) in [0, 1]."""
        if not self.cfg.adaptive_weight:
            return 1.0
        import math
        x = (self.innovation_ema - self.cfg.innovation_threshold) / self.cfg.innovation_scale
        return 1.0 / (1.0 + math.exp(-x))

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _pooled_latents(predicted_latents: torch.Tensor) -> torch.Tensor:
        """Pool patch dim if present, returning (BS, T, d) latents.

        Matches CLFAugmentedCost._pooled_latents (clf_cost.py:101) to stay
        consistent with the Jacobian computed by fit_local_linear_model.py.
        """
        if predicted_latents.dim() == 4:
            return predicted_latents.mean(dim=2)
        if predicted_latents.dim() == 3:
            return predicted_latents
        raise ValueError(
            "predicted_latents must be (BS, T, P, d) or (BS, T, d); "
            f"got shape {tuple(predicted_latents.shape)}"
        )

    def _ensure_predicted(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
        *,
        store_in_cache: bool = True,   # accepted for API stability; unused
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run base.get_cost to populate predicted latents.

        NOTE: a single-use cache between get_cost and get_constraints was
        attempted to halve GPU memory, but reusing the same `predicted` tensor
        in both branches of the augmented Lagrangian loss confuses the autograd
        graph under the real DINOWM (multiple consumers of the same saved
        activations).  Reverting to a fresh rollout per call until a safer
        memory optimization is in place (gradient checkpointing or a single
        cost+constraints entrypoint).
        """
        base_cost = self.base.get_cost(info_dict, action_candidates)
        if "predicted_pixels_embed" not in info_dict:
            raise RuntimeError(
                "base_cost_model did not populate 'predicted_pixels_embed' in "
                "info_dict; StableManifoldConstrainedCost requires the rollout."
            )
        predicted = info_dict["predicted_pixels_embed"]
        return base_cost, predicted

    # -- Costable API ---------------------------------------------------------

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Pixel-MSE goal cost + weighted latent Lyapunov terminal cost V_L(v_H).

        V_L(v) = v^T P_lyap v where P_lyap solves the discrete Lyapunov equation
        for the model's linearization at the touchdown equilibrium.  This is a
        probe-free signal — it operates directly on the latent geometry and
        sidesteps the probe-error compounding that broke the original monitor
        (rho=0.056 in the handover).  Identity-weighted ||z_H - z_*||^2 (v2-v4
        cost) ignores the Lyapunov structure; P_lyap up-weights slow-contracting
        directions and de-weights fast ones, giving a geometrically faithful pull
        toward the stable manifold.

        Args:
            info_dict: planner info; will be mutated to cache predicted latents.
            action_candidates: (B, S, H, D) actions, requires_grad=True for
                LagrangianSolver.

        Returns:
            (B, S) tensor with the same shape contract as CEMSolver/LagrangianSolver.
        """
        base_cost, predicted = self._ensure_predicted(info_dict, action_candidates)
        z_traj = self._pooled_latents(predicted)              # (BS, T, d)
        z_H = z_traj[:, -1, :]                                # (BS, d)
        v_H = z_H - self.z_star                               # (BS, d)
        # V_L(v_H) = v_H^T P_lyap v_H
        V_L_H = (v_H @ self.P_lyap * v_H).sum(dim=-1)         # (BS,)
        effective_weight = self.cfg.cost_weight * self.activation_weight()
        return base_cost + effective_weight * V_L_H.reshape(base_cost.shape)

    def get_constraints(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Stack of inequality constraints g ≤ 0, shape (B, S, C).

        LagrangianSolver (lagrangian.py:178) handles each column with its own
        Lagrange multiplier via dual ascent.  Dimensions vary by toggle:
          C1: T cols       (one per rollout step)
          C2: 1 col        (terminal V_L(v_H) <= alpha * V_L(v_0))
          C3: T-1 cols     (one per transition)
        Total C = sum over enabled constraints.

        Reads from the single-use cache populated by ``get_cost`` to avoid a
        second forward pass through the predictor.  Does not write the cache
        because dual-ascent may call this method outside any get_cost pairing.
        """
        _, predicted = self._ensure_predicted(
            info_dict, action_candidates, store_in_cache=False,
        )
        z_traj = self._pooled_latents(predicted)         # (BS, T, d)
        v_traj = z_traj - self.z_star                     # (BS, T, d)

        BS = action_candidates.shape[0] * action_candidates.shape[1]
        current_bs, num_samples = action_candidates.shape[0], action_candidates.shape[1]

        rows: list[torch.Tensor] = []

        # C1: unstable-subspace energy at every step.
        if self.cfg.use_C1_unstable_bound:
            v_u = v_traj @ self.P_u.T                     # (BS, T, d)
            energy_u = (v_u * v_u).sum(dim=-1)            # (BS, T)
            c1 = energy_u - self.cfg.eps_u ** 2           # (BS, T)
            rows.append(c1)

        # C2: terminal Lyapunov decrease over the full horizon.
        # V_L(v_H) <= alpha * V_L(v_0).
        # Per-step formulation (V_L(v_{t+1}) <= alpha * V_L(v_t) for all t) is
        # infeasible far from z_* because the nonlinear dynamics can temporarily
        # increase V_L before contracting.  The Stable Manifold Theorem only needs
        # the terminal decrease; per-step monotonicity is not required by the proof.
        if self.cfg.use_C2_latent_lyap:
            V_L = (v_traj @ self.P_lyap * v_traj).sum(dim=-1)  # (BS, T)
            # Single terminal constraint: V_L(v_H) - alpha * V_L(v_0) <= 0
            c2 = V_L[:, -1:] - self.cfg.alpha * V_L[:, :1]      # (BS, 1)
            rows.append(c2)

        # C3: decoded-V descent (existing monitor signal, promoted to constraint).
        if self.cfg.use_C3_decoded_lyap:
            T = z_traj.shape[1]
            d = z_traj.shape[2]
            x_hat = self.decoder(z_traj.reshape(BS * T, d))
            V_dec = self.compute_V(x_hat).reshape(BS, T)       # (BS, T)
            if self.cfg.normalized_descent:
                # Δ = (V_t − V_{t+1}) / (V_t + eps); violation ⇔ Δ < 0
                # As a constraint g ≤ 0: -Δ + η ≤ 0
                delta = (V_dec[:, :-1] - V_dec[:, 1:]) / (
                    V_dec[:, :-1].abs() + self.cfg.eps
                )
                c3 = -delta + self.cfg.eta
            else:
                c3 = V_dec[:, 1:] - V_dec[:, :-1] + self.cfg.eta  # (BS, T-1)
            # Optionally scale C3 violations by the adaptive activation.
            # When activation->0 (easy: model accurate), C3 becomes effectively
            # nonbinding, recovering pure CEM/V_L behavior.  When activation->1
            # (extreme: model fails), full constraint enforcement.
            if self.cfg.gate_c3_by_innovation:
                c3 = c3 * self.activation_weight()
            rows.append(c3)

        if not rows:
            # No active constraints — return empty (BS, 0) tensor.
            constraints_flat = torch.zeros(
                BS, 0, device=self.device, dtype=z_traj.dtype
            )
        else:
            # Each row is (BS, K_i); concat along K then reshape (BS, total).
            constraints_flat = torch.cat(rows, dim=-1)        # (BS, total_C)

        return constraints_flat.reshape(current_bs, num_samples, -1)

    # -- Diagnostics ----------------------------------------------------------

    def diagnose_plan(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> dict:
        """Return per-step constraint values for a single plan.

        For logging and paper figures.  Mirrors clf_cost.CLFAugmentedCost.diagnose_plan
        so the eval harness can use either drop-in.
        """
        info = dict(info_dict)
        with torch.no_grad():
            constraints = self.get_constraints(info, action_candidates)
            base_predicted = info.get("predicted_pixels_embed")
            z_traj = (
                self._pooled_latents(base_predicted)
                if base_predicted is not None
                else None
            )
            v_traj = z_traj - self.z_star if z_traj is not None else None

            out: dict = {
                "available": True,
                "spectral_path": self._spectral_path,
                "eps_u": float(self.cfg.eps_u),
                "alpha": float(self.cfg.alpha),
                "eta": float(self.cfg.eta),
                "n_s": int(self.n_s),
                "n_u": int(self.n_u),
                "gap": float(self.gap),
                "spectral_radius_stable": float(self.spectral_radius_stable),
                "constraints": [
                    [float(x) for x in row]
                    for row in constraints.reshape(-1, constraints.shape[-1]).cpu().tolist()
                ],
            }

            if v_traj is not None:
                energy_u_traj = (
                    (v_traj @ self.P_u.T) ** 2
                ).sum(dim=-1).cpu().tolist()
                V_L_traj = (
                    v_traj @ self.P_lyap * v_traj
                ).sum(dim=-1).cpu().tolist()
                out["energy_u_trace"] = energy_u_traj
                out["V_L_trace"] = V_L_traj

        return out

    # -- Module-like glue -----------------------------------------------------

    def eval(self):
        """Propagate to base and decoder."""
        if hasattr(self.base, "eval"):
            self.base.eval()
        self.decoder.eval()
        return self

    def to(self, device):
        """Move everything to ``device``."""
        if hasattr(self.base, "to"):
            self.base.to(device)
        self.decoder.to(device)
        self.device = torch.device(device)
        self.z_star = self.z_star.to(self.device)
        self.P_s = self.P_s.to(self.device)
        self.P_u = self.P_u.to(self.device)
        self.P_lyap = self.P_lyap.to(self.device)
        return self
