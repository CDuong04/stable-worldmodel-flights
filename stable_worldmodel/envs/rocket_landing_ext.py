"""Extended Rocket Landing Environment."""

from __future__ import annotations

from typing import Any
from stable_worldmodel.envs import register
import gymnasium
import numpy as np
import pybullet as p

from stable_worldmodel.envs.pyflyt_rocketlanding import RocketLandingEnv
from stable_worldmodel.perturbations import (
    OUWind, StepGust, LateralKick,
    WIND_PRESETS, LATERAL_KICK_PRESETS,
)

PERTURBATION_PRESETS = {
    "easy": {
        "offset_range": 3.0,
        "tilt_range": 0.05,
        "lat_vel_range": 1.0,
        "vert_vel_range": (-30.0, -15.0),
        "ang_vel_range": 0.02,
    },
    "medium": {
        "offset_range": 8.0,
        "tilt_range": 0.10,
        "lat_vel_range": 2.0,
        "vert_vel_range": (-50.0, -25.0),
        "ang_vel_range": 0.05,
    },
    "hard": {
        "offset_range": 15.0,
        "tilt_range": 0.20,
        "lat_vel_range": 4.0,
        "vert_vel_range": (-70.0, -35.0),
        "ang_vel_range": 0.10,
    },
    "extreme": {
        "offset_range": 25.0,
        "tilt_range": 0.35,
        "lat_vel_range": 6.0,
        "vert_vel_range": (-90.0, -45.0),
        "ang_vel_range": 0.20,
    },
}


class OUPadMotion:
    """Ornstein-Uhlenbeck process for landing pad motion."""

    def __init__(self, theta=0.5, sigma_xy=1.0, sigma_z=0.3,
                 max_radius=8.0, dt=1.0/40.0, seed=None):
        self.theta = theta
        self.sigma_xy = sigma_xy
        self.sigma_z = sigma_z
        self.max_radius = max_radius
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

    def step(self):
        """Advance the OU process by one timestep."""
        noise = self.rng.normal(0, 1, 3)
        noise[:2] *= self.sigma_xy
        noise[2] *= self.sigma_z
        self.velocity += (-self.theta * self.velocity + noise) * self.dt
        self.position += self.velocity * self.dt
        r = np.linalg.norm(self.position[:2])
        if r > self.max_radius:
            self.position[:2] *= self.max_radius / r
            radial = self.position[:2] / (r + 1e-9)
            proj = np.dot(self.velocity[:2], radial)
            if proj > 0:
                self.velocity[:2] -= proj * radial
        self.position[2] = np.clip(self.position[2], -0.3, 0.8)
        return self.position.copy(), self.velocity.copy()

    def reset(self, seed=None):
        """Reset the process state to zero."""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        if seed is not None:
            self.rng = np.random.default_rng(seed)


class RocketLandingExtEnv(RocketLandingEnv):
    """RocketLandingEnv with perturbation presets and moving platform."""

    def __init__(self, ceiling: float = 250.0, **kwargs):
        super().__init__(ceiling=ceiling, **kwargs)
        self.ou_pad = None
        self.moving_pad_enabled = False
        self._perturbation_rng = np.random.default_rng()
        # In-flight wind / kick disturbances (off by default).
        self.wind = None
        self.gust = None
        self.lateral_kick = None
        self._wind_log: list[np.ndarray] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment with optional perturbation and moving pad settings."""
        if options is None:
            options = {}

        perturbation_level = options.pop("perturbation_level", None)
        moving_pad = options.pop("moving_pad", False)
        pad_sigma_xy = options.pop("pad_sigma_xy", 1.0)
        pad_sigma_z = options.pop("pad_sigma_z", 0.3)
        pad_theta = options.pop("pad_theta", 0.5)
        pad_max_radius = options.pop("pad_max_radius", 8.0)
        wind_level = options.pop("wind_level", None)         # "calm"|"light"|"gust"|"storm"
        gust_enabled = options.pop("gust_enabled", False)
        lateral_kick_level = options.pop("lateral_kick_level", None)  # "light"|"moderate"|"severe"

        self._perturbation_rng = np.random.default_rng(seed)
        rng = self._perturbation_rng

        if perturbation_level is not None:
            preset = PERTURBATION_PRESETS.get(perturbation_level)
            if preset is None:
                raise ValueError(
                    f"Unknown perturbation_level '{perturbation_level}'. "
                    f"Choose from: {list(PERTURBATION_PRESETS.keys())}"
                )

            offset_x = rng.uniform(-preset["offset_range"], preset["offset_range"])
            offset_y = rng.uniform(-preset["offset_range"], preset["offset_range"])
            tilt_x = rng.uniform(-preset["tilt_range"], preset["tilt_range"])
            tilt_y = rng.uniform(-preset["tilt_range"], preset["tilt_range"])
            lat_vel_x = rng.uniform(-preset["lat_vel_range"], preset["lat_vel_range"])
            lat_vel_y = rng.uniform(-preset["lat_vel_range"], preset["lat_vel_range"])
            vert_vel = rng.uniform(preset["vert_vel_range"][0], preset["vert_vel_range"][1])
            ang_vel = rng.uniform(-preset["ang_vel_range"], preset["ang_vel_range"], size=3)

            height = rng.uniform(self.ceiling * 0.7, self.ceiling * 0.95)
            self.start_pos = np.array([[offset_x, offset_y, height]])
            self.start_orn = np.array([[tilt_x, tilt_y, 0.0]])

            options["randomize_drop"] = False
            options["accelerate_drop"] = False

            self._ext_lin_vel = np.array([lat_vel_x, lat_vel_y, vert_vel])
            self._ext_ang_vel = ang_vel.copy()

            options.setdefault("starting_fuel_ratio", 0.20)

        else:
            self._ext_lin_vel = None
            self._ext_ang_vel = None
            options.setdefault("randomize_drop", False)
            options.setdefault("accelerate_drop", True)

        obs, info = super().reset(seed=seed, options=options)

        if self._ext_lin_vel is not None:
            rocket_id = self.env.drones[0].Id
            client = self.env._client
            p.resetBasePositionAndOrientation(
                rocket_id,
                self.start_pos[0].tolist(),
                p.getQuaternionFromEuler(self.start_orn[0].tolist()),
                physicsClientId=client,
            )
            p.resetBaseVelocity(
                rocket_id,
                self._ext_lin_vel.tolist(),
                self._ext_ang_vel.tolist(),
                physicsClientId=client,
            )
            self.env.drones[0].update_state()
            self.compute_state()
            obs = self.state

        self.moving_pad_enabled = moving_pad
        if moving_pad:
            self.ou_pad = OUPadMotion(
                theta=pad_theta, sigma_xy=pad_sigma_xy, sigma_z=pad_sigma_z,
                max_radius=pad_max_radius,
                dt=1.0 / self.agent_hz if hasattr(self, "agent_hz") else 1.0 / 40.0,
                seed=seed,
            )
        else:
            self.ou_pad = None

        self._pad_velocity = np.zeros(3)
        self._pad_position = self.landing_pad_position.copy()
        self._perturbation_level = perturbation_level or "default"

        # Initialise in-flight disturbances
        if wind_level is not None:
            wind_cfg = WIND_PRESETS.get(wind_level)
            if wind_cfg is None:
                raise ValueError(
                    f"Unknown wind_level '{wind_level}'. "
                    f"Choose from: {list(WIND_PRESETS.keys())}"
                )
            self.wind = OUWind(cfg=wind_cfg, seed=seed)
        else:
            self.wind = None

        self.gust = StepGust(seed=seed) if gust_enabled else None

        if lateral_kick_level is not None:
            kick_cfg = LATERAL_KICK_PRESETS.get(lateral_kick_level)
            if kick_cfg is None:
                raise ValueError(
                    f"Unknown lateral_kick_level '{lateral_kick_level}'. "
                    f"Choose from: {list(LATERAL_KICK_PRESETS.keys())}"
                )
            self.lateral_kick = LateralKick(cfg=kick_cfg, seed=seed)
        else:
            self.lateral_kick = None

        self._wind_log = []
        # Don't stash disturbance config in `info` — the env wrapper chain
        # enforces that info keys are injected by the wrappers themselves.

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment with optional moving pad + wind disturbance."""
        if self.moving_pad_enabled and self.ou_pad is not None:
            pad_pos, pad_vel = self.ou_pad.step()
            base_pad_pos = np.array([0.0, 0.0, 0.1])
            new_pad_pos = base_pad_pos + pad_pos
            p.resetBasePositionAndOrientation(
                self.landing_pad_id,
                new_pad_pos.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.env._client,
            )
            self.landing_pad_position = pad_pos.copy()
            self.landing_pad_position[2] = 0.0
        else:
            pad_vel = np.zeros(3)

        # Apply wind / gust / kick force in world frame at the rocket centre
        # of mass *before* the physics step. PyBullet integrates over dt.
        F_wind = np.zeros(3)
        if self.wind is not None:
            F_wind = F_wind + self.wind.step()
        if self.gust is not None:
            F_wind = F_wind + self.gust.step()
        if self.lateral_kick is not None:
            F_wind = F_wind + self.lateral_kick.step()
        if np.linalg.norm(F_wind) > 0:
            rocket_id = self.env.drones[0].Id
            p.applyExternalForce(
                objectUniqueId=rocket_id,
                linkIndex=-1,
                forceObj=F_wind.tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=p.LINK_FRAME if False else p.WORLD_FRAME,
                physicsClientId=self.env._client,
            )
            self._wind_log.append(F_wind.copy())

        obs, reward, terminated, truncated, info = super().step(action)

        self._pad_velocity = pad_vel
        self._pad_position = self.landing_pad_position.copy()
        # Avoid adding new info keys — upstream wrapper asserts no collision.

        return obs, reward, terminated, truncated, info

register(
    id="swm/PFRocketLandingExt-v0",
    entry_point="stable_worldmodel.envs.rocket_landing_ext:RocketLandingExtEnv",
)
