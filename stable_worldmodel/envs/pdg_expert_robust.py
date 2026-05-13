"""Robust PDG variants for collecting successful trajectories under disturbance.

Implements four wind-rejection schemes, each wrapping the existing
PDGController:

  1. OracleWindPDG       — feedforward wind compensation using ground-truth
                           wind force from env.unwrapped.{wind, gust}.
  2. ESOWindPDG          — Extended State Observer estimates wind from
                           velocity-derivative residuals; feedforward.
  3. L1AdaptivePDG       — L1 adaptive augmentation on top of PDG attitude
                           command, adapts to lumped disturbance.
  4. SCPLikePDG          — single-iteration SCP wrapper (PDG nominal +
                           explicit wind term in the attitude target).

All variants expose a `compute_control(state, env=None)` method returning the
7-D action. The `env` argument is optional; passing it lets variants 1, 2, 4
read disturbance state directly from the simulator (oracle access).

Usage:
    from stable_worldmodel.envs.pdg_expert_robust import OracleWindPDG
    expert = OracleWindPDG()
    action = expert.compute_control(state, env=env)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stable_worldmodel.envs.pdg_expert import (
    PDGController, PDGParams, parse_obs, quat_to_rotmat,
)


# ----------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------

def world_to_body(quat_wxyz: np.ndarray, vec_world: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into body frame."""
    R_wb = quat_to_rotmat(quat_wxyz)  # body -> world
    return R_wb.T @ vec_world


def wind_force_from_env(env) -> np.ndarray:
    """Read the current world-frame disturbance force from an Ext env.

    Sums OUWind + StepGust + LateralKick if present. Returns zero-vector
    if env doesn't expose disturbance state.
    """
    if env is None:
        return np.zeros(3)
    u = getattr(env, "unwrapped", env)
    F = np.zeros(3)
    for name in ("wind", "gust", "lateral_kick"):
        obj = getattr(u, name, None)
        if obj is not None and hasattr(obj, "F"):
            F = F + np.asarray(obj.F, dtype=float)
    return F


# ----------------------------------------------------------------------
# 1. Oracle wind feedforward
# ----------------------------------------------------------------------

class OracleWindPDG(PDGController):
    """PDG augmented with oracle wind feedforward.

    Reads ground-truth wind from env, converts to body-frame thrust
    compensation via the wind-triangle method. Adds a small attitude tilt
    so that the gimballed thrust cancels the lateral wind force during the
    burn phase.
    """

    def compute_control(self, state, env=None):
        action = super().compute_control(state).copy()
        # action layout: [finx, finy, finz, engine_on, throttle, gimbal_x, gimbal_y]
        engine_on = float(action[3])
        if engine_on < 0.5:
            return action  # coast: gimbal has no thrust to deflect

        F_wind = wind_force_from_env(env)
        if np.linalg.norm(F_wind) < 1e-6:
            return action

        # Convert wind to body-frame; we want thrust to push opposite to wind.
        F_body = world_to_body(state["quaternion"], F_wind)

        throttle = float(action[4])
        # Effective thrust available for gimbal-direction force this step:
        eff_thrust = self.p.max_thrust * (
            self.p.min_frac + (1 - self.p.min_frac) * throttle
        )
        # Gimbal angle needed so horizontal thrust = -F_wind_body horizontal
        # (small-angle approx: sin(theta) ~ theta for small tilts).
        # Action's gimbal_x / gimbal_y are normalized to [-1, 1] via
        # PDGParams.gimbal_limit.
        # Body frame: x-forward, y-left, z-up (standard aerospace).
        # Lateral wind on body x -> need gimbal_y; body y -> gimbal_x.
        # Sign convention is empirical; flip if the rocket diverges.
        d_gimbal_x = -F_body[1] / max(eff_thrust, 1e-3) / self.p.gimbal_limit
        d_gimbal_y = +F_body[0] / max(eff_thrust, 1e-3) / self.p.gimbal_limit
        # Cap the augmentation at a fraction of full deflection so PDG's
        # nominal attitude command isn't drowned out.
        cap = 0.30
        d_gimbal_x = float(np.clip(d_gimbal_x, -cap, cap))
        d_gimbal_y = float(np.clip(d_gimbal_y, -cap, cap))
        action[5] = float(np.clip(action[5] + d_gimbal_x, -1.0, 1.0))
        action[6] = float(np.clip(action[6] + d_gimbal_y, -1.0, 1.0))
        return action


# ----------------------------------------------------------------------
# 2. Extended State Observer wind estimation + feedforward
# ----------------------------------------------------------------------

@dataclass
class ESOConfig:
    """Linear ESO parameters (single bandwidth omega)."""
    omega: float = 4.0    # observer bandwidth (rad/s); higher = faster but noisier
    mass: float = 549.1   # rocket mass (kg, dry + full fuel ~138 + 411)
    g: float = 9.81
    dt: float = 1.0 / 40.0


class ESOWindPDG(PDGController):
    """PDG + linear ESO that estimates lumped lateral disturbance from
    velocity-derivative residuals. No oracle access required.

    State: [v_xy, F_dist_xy] (4-D; we only estimate horizontal disturbance,
    vertical is dominated by gravity + thrust which PDG handles).

    Standard 2nd-order ESO with poles at -omega.
    """

    def __init__(self, params: PDGParams = None, eso_cfg: ESOConfig = None):
        super().__init__(params)
        self.eso_cfg = eso_cfg or ESOConfig()
        self.F_est = np.zeros(2, dtype=float)   # x,y disturbance force
        self.v_est = np.zeros(2, dtype=float)
        self._initialized = False

    def reset(self):
        super().reset()
        self.F_est[:] = 0.0
        self.v_est[:] = 0.0
        self._initialized = False

    def compute_control(self, state, env=None):
        c = self.eso_cfg
        # PDG computes thrust+gimbal nominal action.
        action = super().compute_control(state).copy()

        # Observed horizontal velocity in WORLD frame.
        R_wb = quat_to_rotmat(state["quaternion"])
        v_world = R_wb @ state["velocity"]
        v_xy_obs = v_world[:2]

        if not self._initialized:
            self.v_est[:] = v_xy_obs
            self._initialized = True

        # Known control acceleration estimate (rough: thrust frac * max_thrust * thrust dir / mass)
        engine_on = action[3]
        throttle = action[4]
        thrust_mag = (engine_on > 0.5) * self.p.max_thrust * (
            self.p.min_frac + (1 - self.p.min_frac) * throttle
        )
        thrust_dir_body = np.array([action[5], action[6], 1.0])
        thrust_dir_body /= max(np.linalg.norm(thrust_dir_body), 1e-6)
        thrust_world = R_wb @ (thrust_mag * thrust_dir_body)
        a_known_xy = thrust_world[:2] / c.mass    # gravity is vertical; doesn't enter horizontal eqn

        # ESO innovation: actual v_xy vs. predicted from known dynamics.
        innov = v_xy_obs - self.v_est
        beta1 = 2 * c.omega
        beta2 = c.omega ** 2
        # State updates: v_dot = a_known + F_est/m + beta1 * innov
        #                F_dot = beta2 * m * innov
        self.v_est = self.v_est + c.dt * (a_known_xy + self.F_est / c.mass + beta1 * innov)
        self.F_est = self.F_est + c.dt * (beta2 * c.mass * innov)
        # Hard cap to prevent runaway under saturating inputs.
        self.F_est = np.clip(self.F_est, -50.0, 50.0)

        # Feedforward: convert -F_est into gimbal deflection (body frame).
        if engine_on > 0.5:
            F_world = np.array([self.F_est[0], self.F_est[1], 0.0])
            F_body = R_wb.T @ F_world
            d_gx = -F_body[1] / max(thrust_mag, 1e-3) / self.p.gimbal_limit
            d_gy = +F_body[0] / max(thrust_mag, 1e-3) / self.p.gimbal_limit
            cap = 0.30
            action[5] = float(np.clip(action[5] + np.clip(d_gx, -cap, cap), -1.0, 1.0))
            action[6] = float(np.clip(action[6] + np.clip(d_gy, -cap, cap), -1.0, 1.0))
        return action


# ----------------------------------------------------------------------
# 3. L1 adaptive augmentation
# ----------------------------------------------------------------------

@dataclass
class L1Config:
    """L1 adaptive controller configuration."""
    Gamma: float = 200.0   # adaptation gain (high)
    omega_filter: float = 8.0   # low-pass filter bandwidth (Hz, rad/s actually)
    mass: float = 549.1
    dt: float = 1.0 / 40.0


class L1AdaptivePDG(PDGController):
    """PDG + L1 adaptive augmentation on horizontal acceleration command.

    Reference model: predicted v_xy from PDG nominal command.
    Adaptive law: F_hat += dt * Gamma * (v_xy_obs - v_xy_pred).
    Filter: low-pass at omega_filter to bound the adaptation bandwidth.
    Augmentation: subtract filtered F_hat / mass from acceleration command.
    """

    def __init__(self, params: PDGParams = None, l1_cfg: L1Config = None):
        super().__init__(params)
        self.l1_cfg = l1_cfg or L1Config()
        self.F_hat = np.zeros(2, dtype=float)
        self.F_filt = np.zeros(2, dtype=float)
        self.v_pred = np.zeros(2, dtype=float)
        self._initialized = False

    def reset(self):
        super().reset()
        self.F_hat[:] = 0.0
        self.F_filt[:] = 0.0
        self.v_pred[:] = 0.0
        self._initialized = False

    def compute_control(self, state, env=None):
        c = self.l1_cfg
        action = super().compute_control(state).copy()
        R_wb = quat_to_rotmat(state["quaternion"])
        v_world = R_wb @ state["velocity"]
        v_xy_obs = v_world[:2]

        if not self._initialized:
            self.v_pred[:] = v_xy_obs
            self._initialized = True

        # Known control acceleration (similar to ESO)
        engine_on = action[3]
        throttle = action[4]
        thrust_mag = (engine_on > 0.5) * self.p.max_thrust * (
            self.p.min_frac + (1 - self.p.min_frac) * throttle
        )
        thrust_dir_body = np.array([action[5], action[6], 1.0])
        thrust_dir_body /= max(np.linalg.norm(thrust_dir_body), 1e-6)
        a_known_xy = (R_wb @ (thrust_mag * thrust_dir_body))[:2] / c.mass

        # Reference model integration: v_pred += dt * (a_known + F_filt/m)
        self.v_pred = self.v_pred + c.dt * (a_known_xy + self.F_filt / c.mass)

        # Adaptation: F_hat -= dt * Gamma * (v_pred - v_obs)
        self.F_hat = self.F_hat - c.dt * c.Gamma * (self.v_pred - v_xy_obs)
        self.F_hat = np.clip(self.F_hat, -50.0, 50.0)

        # Low-pass filter (1st order, omega_filter rad/s)
        alpha = c.dt * c.omega_filter / (1 + c.dt * c.omega_filter)
        self.F_filt = (1 - alpha) * self.F_filt + alpha * self.F_hat

        # Augment: subtract F_filt-induced gimbal correction
        if engine_on > 0.5:
            F_world = np.array([self.F_filt[0], self.F_filt[1], 0.0])
            F_body = R_wb.T @ F_world
            d_gx = -F_body[1] / max(thrust_mag, 1e-3) / self.p.gimbal_limit
            d_gy = +F_body[0] / max(thrust_mag, 1e-3) / self.p.gimbal_limit
            cap = 0.30
            action[5] = float(np.clip(action[5] + np.clip(d_gx, -cap, cap), -1.0, 1.0))
            action[6] = float(np.clip(action[6] + np.clip(d_gy, -cap, cap), -1.0, 1.0))
        return action


# ----------------------------------------------------------------------
# 4. Single-iteration SCP-like wrapper
# ----------------------------------------------------------------------

class SCPLikePDG(OracleWindPDG):
    """Single-step SCP approximation: PDG provides nominal trajectory;
    we add an attitude-target perturbation that pre-tilts the rocket into
    the wind by the steady-state lean angle.

    For now this aliases OracleWindPDG (since PDG already does the closed-loop
    re-planning at every step). True SCP would batch-solve over the horizon
    with wind in the dynamics; that's a re-derivation of the SOCP.
    """
    pass


# ----------------------------------------------------------------------
# Variant registry for easy selection
# ----------------------------------------------------------------------

VARIANTS = {
    "oracle_wind": OracleWindPDG,
    "eso":         ESOWindPDG,
    "l1":          L1AdaptivePDG,
    "scp":         SCPLikePDG,
}


def make_expert(variant: str, params: PDGParams = None):
    """Factory: returns an instance of the requested variant."""
    cls = VARIANTS.get(variant)
    if cls is None:
        raise ValueError(f"Unknown variant '{variant}'. Choices: {list(VARIANTS)}")
    return cls(params)
