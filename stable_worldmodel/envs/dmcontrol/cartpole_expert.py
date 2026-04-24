"""Analytic expert policy for DMControl cart-pole swing-up and balance."""

from __future__ import annotations

import numpy as np

from stable_worldmodel.policy import BasePolicy


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _to_batched_2d(array: np.ndarray) -> np.ndarray:
    """Normalize env info arrays to shape (num_envs, dim)."""
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 1:
        return array[None, :]

    # World infos often carry a singleton history dimension: (E, 1, D).
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, 0, :]

    if array.ndim == 2:
        return array

    raise ValueError(f'Unsupported cartpole info shape: {array.shape}')


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
        balance_angle_threshold: float = 1.3,
        balance_angular_velocity_threshold: float = 12.0,
        balance_k_theta: float = 18.0,
        balance_k_theta_dot: float = 5.0,
        balance_k_x: float = 1.5,
        balance_k_x_dot: float = 4.0,
        balance_force_gains: tuple[float, float, float, float]
        | None = (-3.53553391, -52.23926733, -6.0104904, -12.46748597),
        swing_k_energy: float = 0.10,
        swing_k_x: float = 0.08,
        swing_k_x_dot: float = 0.12,
        swing_k_theta: float = 0.0,
        kick_gain: float = 2.5,
        kick_angle_threshold: float = 0.35,
        kick_angular_velocity_threshold: float = 0.35,
        gravity: float = 9.81,
        pole_length: float = 0.5,
        motor_force: float = 10.0,
        force_scale: float = 1.0,
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
        self.balance_force_gains = (
            None
            if balance_force_gains is None
            else np.asarray(balance_force_gains, dtype=np.float32)
        )
        self.swing_k_energy = float(swing_k_energy)
        self.swing_k_x = float(swing_k_x)
        self.swing_k_x_dot = float(swing_k_x_dot)
        self.swing_k_theta = float(swing_k_theta)
        self.kick_gain = float(kick_gain)
        self.kick_angle_threshold = float(kick_angle_threshold)
        self.kick_angular_velocity_threshold = float(
            kick_angular_velocity_threshold
        )
        self.gravity = float(gravity)
        self.pole_length = float(pole_length)
        self.motor_force = float(motor_force)
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
        if self.balance_force_gains is not None:
            state = np.stack([x, theta, x_dot, theta_dot], axis=1)
            force = -(state * self.balance_force_gains[None, :]).sum(axis=1)
            return force / self.motor_force

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
        # theta=0 is upright and theta=pi is hanging down in DMControl.
        target_energy = 2.0 * self.gravity * self.pole_length
        energy = (
            0.5 * (self.pole_length * theta_dot) ** 2
            + self.gravity * self.pole_length * (1.0 + np.cos(theta))
        )
        energy_gap = target_energy - energy

        # For this angle convention, the pump sign must oppose theta_dot*cos(theta)
        # to add energy while the pole is below the top.
        pump = (
            -self.swing_k_energy
            * theta_dot
            * np.cos(theta)
            * energy_gap
        )
        center = -self.swing_k_x * x - self.swing_k_x_dot * x_dot
        shape = self.swing_k_theta * np.sin(theta)

        # The exact swing-up start is nearly motionless, so pure energy shaping
        # can stall. Give it an initial shove to break symmetry.
        down_angle = _wrap_to_pi(theta - np.pi)
        needs_kick = (np.abs(down_angle) < self.kick_angle_threshold) & (
            np.abs(theta_dot) < self.kick_angular_velocity_threshold
        )
        kick_dir = -np.sign(np.sin(theta))
        kick_dir = np.where(kick_dir == 0.0, 1.0, kick_dir)
        kick = self.kick_gain * kick_dir

        return np.where(needs_kick, kick + center, pump + center + shape)

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
        if 'qpos' in info_dict and 'qvel' in info_dict:
            qpos = _to_batched_2d(info_dict['qpos'])
            qvel = _to_batched_2d(info_dict['qvel'])

            x = qpos[:, 0]
            theta = _wrap_to_pi(qpos[:, 1])
            x_dot = qvel[:, 0]
            theta_dot = qvel[:, 1]
        elif 'observation' in info_dict:
            obs = _to_batched_2d(info_dict['observation'])
            if obs.shape[1] != 5:
                raise ValueError(
                    f'Expected cartpole observation dim 5, got {obs.shape}'
                )

            x = obs[:, 0]
            cos_theta = np.clip(obs[:, 1], -1.0, 1.0)
            sin_theta = np.clip(obs[:, 2], -1.0, 1.0)
            theta = _wrap_to_pi(np.arctan2(sin_theta, cos_theta))
            x_dot = obs[:, 3]
            theta_dot = obs[:, 4]
        else:
            raise AssertionError(
                "CartpoleExpertPolicy requires 'qpos'/'qvel' or 'observation'"
            )

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
        action = np.clip(
            action_raw / max(self.force_scale, 1e-6), -1.0, 1.0
        ).astype(np.float32)
        return action[:, None]
