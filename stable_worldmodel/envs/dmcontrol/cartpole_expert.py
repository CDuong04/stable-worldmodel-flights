"""Analytic expert policy for DMControl cart-pole swing-up and balance."""

from __future__ import annotations

import numpy as np

from stable_worldmodel.policy import BasePolicy


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class CartpoleExpertPolicy(BasePolicy):
    """Hybrid swing-up and balance controller for cart-pole.

    The controller uses:
    - energy shaping far from the upright equilibrium, following the standard
      swing-up recipe described in the underactuated control literature; and
    - a linear stabilizer near the upright position.

    Optional action noise and short perturbation bursts can be enabled during
    collection to broaden the state-action manifold while remaining near expert
    behavior on average.
    """

    def __init__(
        self,
        balance_angle_threshold: float = 0.32,
        balance_angular_velocity_threshold: float = 1.6,
        balance_k_theta: float = 8.0,
        balance_k_theta_dot: float = 1.6,
        balance_k_x: float = 1.2,
        balance_k_x_dot: float = 1.5,
        swing_k_energy: float = 0.075,
        swing_k_x: float = 0.45,
        swing_k_x_dot: float = 0.55,
        swing_k_theta: float = 0.25,
        gravity: float = 9.81,
        pole_length: float = 1.0,
        force_scale: float = 8.0,
        noise_std: float = 0.0,
        burst_prob: float = 0.0,
        burst_noise_std: float = 0.0,
        burst_steps_range: tuple[int, int] = (2, 6),
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = 'expert'

        self.balance_angle_threshold = float(balance_angle_threshold)
        self.balance_angular_velocity_threshold = float(
            balance_angular_velocity_threshold
        )
        self.balance_k_theta = float(balance_k_theta)
        self.balance_k_theta_dot = float(balance_k_theta_dot)
        self.balance_k_x = float(balance_k_x)
        self.balance_k_x_dot = float(balance_k_x_dot)
        self.swing_k_energy = float(swing_k_energy)
        self.swing_k_x = float(swing_k_x)
        self.swing_k_x_dot = float(swing_k_x_dot)
        self.swing_k_theta = float(swing_k_theta)
        self.gravity = float(gravity)
        self.pole_length = float(pole_length)
        self.force_scale = float(force_scale)
        self.noise_std = float(noise_std)
        self.burst_prob = float(burst_prob)
        self.burst_noise_std = float(burst_noise_std)
        self.burst_steps_range = tuple(int(v) for v in burst_steps_range)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._burst_steps_remaining: np.ndarray | None = None

    def set_env(self, env):
        self.env = env
        self._burst_steps_remaining = np.zeros(self.env.num_envs, dtype=np.int32)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _balance_control(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        x_dot: np.ndarray,
        theta_dot: np.ndarray,
    ) -> np.ndarray:
        return (
            self.balance_k_theta * theta
            + self.balance_k_theta_dot * theta_dot
            - self.balance_k_x * x
            - self.balance_k_x_dot * x_dot
        )

    def _swingup_control(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        x_dot: np.ndarray,
        theta_dot: np.ndarray,
    ) -> np.ndarray:
        # Reparameterize the angle so alpha=0 is the hanging-down state.
        alpha = _wrap_to_pi(theta + np.pi)
        target_energy = 2.0 * self.gravity * self.pole_length
        energy = (
            0.5 * (self.pole_length * theta_dot) ** 2
            + self.gravity * self.pole_length * (1.0 - np.cos(alpha))
        )
        energy_error = energy - target_energy

        return (
            self.swing_k_energy
            * theta_dot
            * np.cos(alpha)
            * energy_error
            - self.swing_k_x * x
            - self.swing_k_x_dot * x_dot
            + self.swing_k_theta * np.sin(theta)
        )

    def _maybe_apply_bursts(self, action_raw: np.ndarray) -> np.ndarray:
        if self._burst_steps_remaining is None or self.burst_prob <= 0.0:
            return action_raw

        inactive = self._burst_steps_remaining <= 0
        if np.any(inactive):
            starts = self.rng.random(inactive.sum()) < self.burst_prob
            if np.any(starts):
                burst_len = self.rng.integers(
                    self.burst_steps_range[0],
                    self.burst_steps_range[1] + 1,
                    size=starts.sum(),
                )
                inactive_idx = np.flatnonzero(inactive)
                self._burst_steps_remaining[inactive_idx[starts]] = burst_len

        active = self._burst_steps_remaining > 0
        if np.any(active):
            if self.burst_noise_std > 0.0:
                action_raw = action_raw.copy()
                action_raw[active] += self.rng.normal(
                    loc=0.0,
                    scale=self.burst_noise_std,
                    size=active.sum(),
                )
            self._burst_steps_remaining[active] -= 1

        return action_raw

    def get_action(self, info_dict, **kwargs):
        assert 'qpos' in info_dict, "CartpoleExpertPolicy requires 'qpos'"
        assert 'qvel' in info_dict, "CartpoleExpertPolicy requires 'qvel'"

        qpos = np.asarray(info_dict['qpos'], dtype=np.float32)
        qvel = np.asarray(info_dict['qvel'], dtype=np.float32)

        if qpos.ndim == 1:
            qpos = qpos[None, :]
        if qvel.ndim == 1:
            qvel = qvel[None, :]

        x = qpos[:, 0]
        theta = _wrap_to_pi(qpos[:, 1])
        x_dot = qvel[:, 0]
        theta_dot = qvel[:, 1]

        balance_action = self._balance_control(x, theta, x_dot, theta_dot)
        swingup_action = self._swingup_control(x, theta, x_dot, theta_dot)

        use_balance = (
            np.abs(theta) < self.balance_angle_threshold
        ) & (
            np.abs(theta_dot) < self.balance_angular_velocity_threshold
        )
        action_raw = np.where(use_balance, balance_action, swingup_action)

        if self.noise_std > 0.0:
            action_raw += self.rng.normal(
                loc=0.0,
                scale=self.noise_std,
                size=action_raw.shape,
            )

        action_raw = self._maybe_apply_bursts(action_raw)
        action = np.tanh(action_raw / self.force_scale).astype(np.float32)
        return action[:, None]
