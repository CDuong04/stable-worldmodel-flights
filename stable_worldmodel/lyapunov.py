"""Decoded Lyapunov monitor for the RocketJEPA paper.

Given a state decoder d_phi: z_t -> hat{x}_t and pad position p_pad, computes:

    V(hat{x}) = || hat{p} - p_pad ||^2 + alpha || hat{v} ||^2 + beta || hat{omega} ||^2
    delta_t  = (V_t - V_{t+1}) / (V_t + eps)

The monitor runs in closed-loop alongside MPC (no MPC dependence). Reports
mean descent rate and violation rate over an episode.

State layout (17-D, matches PFRocketLandingExt obs):
  [0:3]   position (x, y, z)            -> p
  [3:6]   velocity (vx, vy, vz)          -> v
  [6:10]  quaternion (qw, qx, qy, qz)    -> ignored
  [10:13] angular velocity (wx, wy, wz)  -> omega
  [13]    fuel mass                      -> ignored
  [14:17] thrust state                   -> ignored

If your decoder uses a different layout, override `pos_idx`, `vel_idx`,
`omega_idx` at construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class LyapunovConfig:
    alpha: float = 0.5         # velocity weight
    beta: float = 0.1          # angular-velocity weight
    eps: float = 1e-6
    pad_pos: tuple = (0.0, 0.0, 0.0)
    pos_idx: tuple = (0, 1, 2)
    vel_idx: tuple = (3, 4, 5)
    omega_idx: tuple = (10, 11, 12)


class LyapunovMonitor:
    """Tracks V and descent rate over a closed-loop rollout."""

    def __init__(self, decoder, cfg: LyapunovConfig | None = None, device: str = "cpu"):
        self.decoder = decoder.to(device).eval()
        self.cfg = cfg or LyapunovConfig()
        self.device = device
        self.reset()

    def reset(self):
        self._Vs: list[float] = []
        self._V_tp1s: list[float] = []
        self._deltas: list[float] = []

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, embed_dim) or (embed_dim,) -> (B, state_dim) decoded state."""
        z = z.to(self.device).float()
        if z.ndim == 1:
            z = z.unsqueeze(0)
        with torch.no_grad():
            x_hat = self.decoder(z)
        return x_hat

    def compute_V(self, x_hat: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        p = x_hat[..., list(cfg.pos_idx)]
        v = x_hat[..., list(cfg.vel_idx)]
        omega = x_hat[..., list(cfg.omega_idx)]
        pad = torch.as_tensor(cfg.pad_pos, dtype=x_hat.dtype, device=x_hat.device)
        return ((p - pad) ** 2).sum(-1) + cfg.alpha * (v ** 2).sum(-1) + cfg.beta * (omega ** 2).sum(-1)

    def step(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> dict:
        """Record one (z_t, z_{t+1}) transition. Returns dict with V_t, delta_t."""
        x_t  = self.decode(z_t)
        x_tp = self.decode(z_tp1)
        V_t  = self.compute_V(x_t).item()
        V_tp = self.compute_V(x_tp).item()
        delta = (V_t - V_tp) / (V_t + self.cfg.eps)
        self._Vs.append(V_t)
        self._V_tp1s.append(V_tp)
        self._deltas.append(delta)
        return {"V_t": V_t, "V_tp1": V_tp, "delta_t": delta}

    def summary(self) -> dict:
        if not self._deltas:
            return {"mean_descent": 0.0, "violation_rate": 0.0, "n_steps": 0}
        d = np.asarray(self._deltas)
        return {
            "mean_descent": float(d.mean()),
            "violation_rate": float((d < 0).mean()),
            "min_descent": float(d.min()),
            "max_V": float(np.max(self._Vs)),
            "final_V": float(self._Vs[-1]),
            "n_steps": int(len(d)),
        }

    def trace(self) -> dict:
        """Return per-transition monitor traces for downstream calibration."""
        return {
            "V_t_trace": [float(v) for v in self._Vs],
            "V_tp1_trace": [float(v) for v in self._V_tp1s],
            "delta_trace": [float(d) for d in self._deltas],
        }
