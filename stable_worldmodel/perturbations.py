"""In-flight disturbance models for the rocket landing benchmark.

Two classes: `OUWind` (correlated horizontal wind force, aerospace-standard)
and `StepGust` (sudden step-like wind gusts that decay exponentially, matches
FAA gust models). Both produce a 3-D force vector (applied in world frame) at
each environment step.

Theory (why OU wind is the principled default):
    Atmospheric turbulence is conventionally modelled as a zero-mean
    band-limited stochastic process. The simplest model that captures both
    (a) bounded variance and (b) temporal correlation is the
    Ornstein-Uhlenbeck process:

        dw/dt = -theta * w(t) + sigma * eta(t)

    where eta is white noise. This is the discrete analogue of the Dryden
    turbulence model used in aerospace simulation. Unlike white Gaussian
    noise on force, OU is (i) physically realisable, (ii) has closed-form
    mean/variance, (iii) admits a Kalman state-space form with states
    (position, velocity, wind_force) — directly compatible with the paper's
    Gaussian-propagation framework (RocketJEPA Sec. 4.2).

Notes on magnitude ranges:
    The PFRocket benchmark uses ~20 kg vehicle at 40 Hz simulation. A
    comfortable expert-solvable wind field hits body-frame force up to
    ~0.5 * weight = 5 N. Beyond that, the convex PDG expert starts failing;
    that is precisely where the learned world-model + Lyapunov monitor
    should earn its keep.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OUWindConfig:
    """Config for OU wind disturbance.

    theta:           mean-reversion rate (1/s)   — higher = faster decorrelation
    sigma_horizontal: horizontal force scale (N/s^0.5, Ito integration units)
    sigma_vertical:   vertical force scale (typically 1/3 horizontal in Dryden)
    max_force:        hard clamp on |F_wind|   (N, safety rail)
    dt:               integration timestep (env step duration)
    """
    theta: float = 0.3
    sigma_horizontal: float = 1.5
    sigma_vertical: float = 0.5
    max_force: float = 10.0
    dt: float = 1.0 / 40.0


class OUWind:
    """Ornstein-Uhlenbeck wind force in world frame (N)."""

    def __init__(self, cfg: OUWindConfig | None = None, seed: int | None = None):
        self.cfg = cfg or OUWindConfig()
        self.rng = np.random.default_rng(seed)
        self.F = np.zeros(3, dtype=np.float64)

    def step(self) -> np.ndarray:
        c = self.cfg
        sigma = np.array([c.sigma_horizontal, c.sigma_horizontal, c.sigma_vertical])
        # Euler-Maruyama step: dF = -theta F dt + sigma sqrt(dt) dW
        dW = self.rng.standard_normal(3)
        self.F = self.F + (-c.theta * self.F * c.dt) + sigma * np.sqrt(c.dt) * dW
        nrm = np.linalg.norm(self.F)
        if nrm > c.max_force:
            self.F *= c.max_force / nrm
        return self.F.copy()

    def reset(self, seed: int | None = None):
        self.F[:] = 0.0
        if seed is not None:
            self.rng = np.random.default_rng(seed)


@dataclass
class StepGustConfig:
    """Config for step-gust disturbance (FAA-style discrete gusts).

    gust_rate_hz:   expected number of gusts per second (Poisson)
    peak_force:     gust peak magnitude (N)
    decay_time:     exponential decay time-constant (s)
    direction_std:  std of the gust direction angle (rad)
    dt:             env step duration
    """
    gust_rate_hz: float = 0.2
    peak_force: float = 6.0
    decay_time: float = 1.0
    direction_std: float = np.pi / 3
    dt: float = 1.0 / 40.0


class StepGust:
    """Poisson-triggered gusts with exponential decay."""

    def __init__(self, cfg: StepGustConfig | None = None, seed: int | None = None):
        self.cfg = cfg or StepGustConfig()
        self.rng = np.random.default_rng(seed)
        self.F = np.zeros(3, dtype=np.float64)

    def step(self) -> np.ndarray:
        c = self.cfg
        # Poisson trigger
        if self.rng.random() < c.gust_rate_hz * c.dt:
            theta_xy = self.rng.normal(0.0, c.direction_std)
            direction = np.array([np.cos(theta_xy), np.sin(theta_xy), 0.0])
            self.F = c.peak_force * direction
        # exponential decay
        self.F *= np.exp(-c.dt / c.decay_time)
        return self.F.copy()

    def reset(self, seed: int | None = None):
        self.F[:] = 0.0
        if seed is not None:
            self.rng = np.random.default_rng(seed)


# Paper-facing presets to mirror the static-IC PERTURBATION_PRESETS.
WIND_PRESETS = {
    "calm":  OUWindConfig(theta=0.5, sigma_horizontal=0.3, sigma_vertical=0.1, max_force=2.0),
    "light": OUWindConfig(theta=0.4, sigma_horizontal=0.8, sigma_vertical=0.25, max_force=5.0),
    "gust":  OUWindConfig(theta=0.3, sigma_horizontal=1.5, sigma_vertical=0.5, max_force=10.0),
    "storm": OUWindConfig(theta=0.2, sigma_horizontal=3.0, sigma_vertical=1.0, max_force=18.0),
}


@dataclass
class LateralKickConfig:
    """Discrete lateral Δv impulses — simulates a sudden control-surface or
    attitude-thruster failure. Uses a single-step impulse applied at Poisson
    times. The kick is purely horizontal; vertical is left to wind/gust.

    kick_rate_hz:  expected kicks per second (Poisson)
    kick_magnitude: impulse peak force (N) for one env step
    direction_std:  std of the horizontal direction (rad)
    dt:             env step duration
    """
    kick_rate_hz: float = 0.3
    kick_magnitude: float = 15.0
    direction_std: float = np.pi
    dt: float = 1.0 / 40.0


class LateralKick:
    """Poisson-triggered impulsive lateral force (world frame, horizontal only)."""

    def __init__(self, cfg: LateralKickConfig | None = None, seed: int | None = None):
        self.cfg = cfg or LateralKickConfig()
        self.rng = np.random.default_rng(seed)

    def step(self) -> np.ndarray:
        c = self.cfg
        if self.rng.random() < c.kick_rate_hz * c.dt:
            theta_xy = self.rng.uniform(-c.direction_std, c.direction_std)
            return c.kick_magnitude * np.array([np.cos(theta_xy), np.sin(theta_xy), 0.0])
        return np.zeros(3, dtype=np.float64)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)


LATERAL_KICK_PRESETS = {
    "light": LateralKickConfig(kick_rate_hz=0.2, kick_magnitude=8.0),
    "moderate": LateralKickConfig(kick_rate_hz=0.3, kick_magnitude=15.0),
    "severe": LateralKickConfig(kick_rate_hz=0.5, kick_magnitude=25.0),
}
