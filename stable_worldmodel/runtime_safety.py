"""Runtime safety belt — reactive layer of the two-layer stability stack.

While StableManifoldConstrainedCost (search-time, predictive) prevents most
violations *before* an action is taken, this module watches what actually
happens *after* the action is executed.  When realized Lyapunov descent
disagrees with the planner's prediction for several steps in a row, the
RuntimeSafetyMonitor flags the episode as unsafe and (optionally) invokes a
fallback controller — by default a PDG/expert burn from the GNC module.

Pairs with stable_worldmodel.lyapunov.LyapunovMonitor:
  - LyapunovMonitor computes V(decoded x) and per-step descent δ_t.
  - This wrapper adds a rolling window over δ_t, a binary safety flag, and
    a fallback-trigger hook.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from stable_worldmodel.lyapunov import LyapunovMonitor


@dataclass
class RuntimeSafetyConfig:
    """Hyperparameters for the runtime monitor.

    Attributes:
        window_size: number of recent steps over which to compute the
            rolling violation rate.  Default 10 ≈ 0.5 s at 20 Hz control.
        violation_threshold: realized descent δ_t < this counts as a violation.
            Default 0.0 (any non-descent is a violation).
        rate_threshold: rolling violation_rate > this triggers fallback.
            Default 0.4 (40% of recent steps violated).
        min_window: do not trigger fallback before observing at least this many
            steps.  Prevents false triggers in the first few control steps.
        cooldown: number of consecutive in-tolerance steps required to clear
            an active fallback flag.
    """
    window_size: int = 10
    violation_threshold: float = 0.0
    rate_threshold: float = 0.4
    min_window: int = 4
    cooldown: int = 5


class RuntimeSafetyMonitor:
    """Wrap a LyapunovMonitor with violation-rate tracking and fallback.

    Usage:
        monitor = RuntimeSafetyMonitor(
            decoder=probe,
            fallback_fn=pdg_burn_action,
            cfg=RuntimeSafetyConfig(),
        )

        # In the outer control loop:
        action = solver.solve(info)["actions"]
        if monitor.should_fallback(z_t, z_tp1):
            action = monitor.fallback(info, action)
        env.step(action)

    Args:
        decoder: probe d_φ used by the underlying LyapunovMonitor.
        fallback_fn: optional callable (info_dict, planned_action) -> safe_action.
            If None, fallback() returns the planned action unchanged but the
            flag still fires so an outer loop can decide what to do.
        cfg: RuntimeSafetyConfig hyperparameters.
        device: where the monitor's decoder runs.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        fallback_fn: Optional[Callable[[dict, torch.Tensor], torch.Tensor]] = None,
        cfg: Optional[RuntimeSafetyConfig] = None,
        lyapunov_cfg=None,
        device: str = "cpu",
    ):
        self.lyapunov = LyapunovMonitor(decoder, cfg=lyapunov_cfg, device=device)
        self.cfg = cfg or RuntimeSafetyConfig()
        self.fallback_fn = fallback_fn

        self._delta_window: deque[float] = deque(maxlen=self.cfg.window_size)
        self._fallback_active = False
        self._cooldown_counter = 0
        self._n_total_steps = 0
        self._n_total_violations = 0
        self._n_fallback_triggers = 0

    def reset(self):
        """Call at the start of each episode."""
        self.lyapunov.reset()
        self._delta_window.clear()
        self._fallback_active = False
        self._cooldown_counter = 0
        self._n_total_steps = 0
        self._n_total_violations = 0
        self._n_fallback_triggers = 0

    # -- Per-step API --------------------------------------------------------

    def observe(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> dict:
        """Record one (z_t, z_{t+1}) transition; update flags.

        Returns the LyapunovMonitor.step() output augmented with safety state.
        """
        step_info = self.lyapunov.step(z_t, z_tp1)
        delta = float(step_info["delta_t"])
        self._delta_window.append(delta)
        self._n_total_steps += 1

        is_violation = delta < self.cfg.violation_threshold
        if is_violation:
            self._n_total_violations += 1

        # Update fallback state machine
        previously_active = self._fallback_active
        if len(self._delta_window) >= self.cfg.min_window:
            viol_rate = sum(
                1 for d in self._delta_window if d < self.cfg.violation_threshold
            ) / len(self._delta_window)
        else:
            viol_rate = 0.0

        if not self._fallback_active:
            if viol_rate > self.cfg.rate_threshold:
                self._fallback_active = True
                self._cooldown_counter = 0
                self._n_fallback_triggers += 1
        else:
            if not is_violation:
                self._cooldown_counter += 1
                if self._cooldown_counter >= self.cfg.cooldown:
                    self._fallback_active = False
                    self._cooldown_counter = 0
            else:
                self._cooldown_counter = 0

        return {
            **step_info,
            "violation": is_violation,
            "rolling_violation_rate": viol_rate,
            "fallback_active": self._fallback_active,
            "fallback_newly_triggered": (
                self._fallback_active and not previously_active
            ),
        }

    def should_fallback(self) -> bool:
        """Whether the monitor currently recommends switching to fallback."""
        return self._fallback_active

    def fallback(self, info: dict, planned_action: torch.Tensor) -> torch.Tensor:
        """Apply the configured fallback policy to override the planned action."""
        if self.fallback_fn is None or not self._fallback_active:
            return planned_action
        return self.fallback_fn(info, planned_action)

    # -- Diagnostics ---------------------------------------------------------

    def summary(self) -> dict:
        """Episode-level summary for paper figures / logging."""
        base = self.lyapunov.summary()
        return {
            **base,
            "total_violations": int(self._n_total_violations),
            "total_steps": int(self._n_total_steps),
            "violation_rate": (
                self._n_total_violations / max(self._n_total_steps, 1)
            ),
            "fallback_triggers": int(self._n_fallback_triggers),
            "fallback_currently_active": bool(self._fallback_active),
        }

    def trace(self) -> dict:
        """Per-step traces for offline analysis."""
        return self.lyapunov.trace()
