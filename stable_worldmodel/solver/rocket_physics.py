"""Differentiable 6-DOF rocket physics and constraint cost for MPC planning."""

import torch
import torch.nn.functional as F
from torch import nn
import math


def quat_to_rotmat_batch(q: torch.Tensor) -> torch.Tensor:
    """Batched quaternion (wxyz) to 3x3 rotation matrix."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def quat_integrate_batch(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """First-order Euler quaternion integration."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    ox, oy, oz = omega[:, 0:1], omega[:, 1:2], omega[:, 2:3]
    dw = 0.5 * (-x * ox - y * oy - z * oz)
    dx = 0.5 * (w * ox + y * oz - z * oy)
    dy = 0.5 * (w * oy + z * ox - x * oz)
    dz = 0.5 * (w * oz + x * oy - y * ox)
    q_new = torch.cat([w + dw * dt, x + dx * dt, y + dy * dt, z + dz * dt], dim=-1)
    q_new = q_new / (torch.norm(q_new, dim=-1, keepdim=True) + 1e-8)
    return q_new


def tilt_from_quat_batch(q: torch.Tensor) -> torch.Tensor:
    """Tilt angle (body z vs world z) from quaternion (wxyz)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    cos_tilt = torch.clamp(1 - 2 * (x * x + y * y), -1.0, 1.0)
    return torch.acos(cos_tilt)


class RocketPhysicsModel(nn.Module):
    """Differentiable 6-DOF rocket physics propagator."""

    def __init__(
        self,
        max_thrust: float = 7607.0,
        min_thrust_frac: float = 0.20,
        dry_mass: float = 138.2,
        fuel_mass: float = 410.9,
        g: float = 9.81,
        Isp: float = 225.0,
        gimbal_limit: float = 0.0873,
        dt: float = 0.025,
    ):
        super().__init__()
        self.register_buffer("_max_thrust", torch.tensor(max_thrust))
        self.register_buffer("_min_thrust_frac", torch.tensor(min_thrust_frac))
        self.register_buffer("_dry_mass", torch.tensor(dry_mass))
        self.register_buffer("_fuel_mass", torch.tensor(fuel_mass))
        self.register_buffer("_g", torch.tensor(g))
        self.register_buffer("_Isp", torch.tensor(Isp))
        self.register_buffer("_gimbal_limit", torch.tensor(gimbal_limit))
        self.register_buffer("_dt", torch.tensor(dt))

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Propagate (B, 17) state one timestep given (B, 7) action."""
        pos = state[:, 0:3]
        vel = state[:, 3:6]
        quat = state[:, 6:10]
        ang_vel = state[:, 10:13]
        fuel = state[:, 13:14]
        tgt_rel = state[:, 14:17]

        ignition = action[:, 3:4]
        throttle = action[:, 4:5]
        gim_x = action[:, 5:6]
        gim_y = action[:, 6:7]

        mass = self._dry_mass + fuel * self._fuel_mass
        T_mag = ignition * throttle * self._max_thrust

        gim_mag = torch.sqrt(gim_x ** 2 + gim_y ** 2 + 1e-8)
        sin_gim = torch.sin(gim_mag)
        cos_gim = torch.cos(gim_mag)

        body_thrust_x = sin_gim * gim_x / (gim_mag + 1e-8)
        body_thrust_y = sin_gim * gim_y / (gim_mag + 1e-8)
        body_thrust_z = cos_gim
        body_thrust = torch.cat([body_thrust_x, body_thrust_y, body_thrust_z], dim=-1)

        R = quat_to_rotmat_batch(quat)
        world_thrust = torch.bmm(R, body_thrust.unsqueeze(-1)).squeeze(-1)

        gravity = torch.zeros_like(vel)
        gravity[:, 2] = -self._g
        accel = (T_mag / mass) * world_thrust + gravity

        vel_new = vel + accel * self._dt
        pos_new = pos + vel * self._dt

        mdot = T_mag / (self._Isp * self._g)
        fuel_used = mdot * self._dt / self._fuel_mass
        fuel_new = torch.clamp(fuel - fuel_used, min=0.0)

        quat_new = quat_integrate_batch(quat, ang_vel, float(self._dt))
        tgt_rel_new = tgt_rel - (pos_new - pos)

        return torch.cat([pos_new, vel_new, quat_new, ang_vel, fuel_new, tgt_rel_new], dim=-1)

    def rollout(self, state_0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out (B, T, 7) actions from (B, 17) initial state."""
        B, T, _ = actions.shape
        states = [state_0]
        state = state_0
        for t in range(T):
            state = self.step(state, actions[:, t])
            states.append(state)
        return torch.stack(states, dim=1)


class RocketConstraintCost(nn.Module):
    """Soft constraint violation penalties over a state trajectory."""

    def __init__(
        self,
        w_ground: float = 50.0,
        w_fuel: float = 20.0,
        w_glide: float = 5.0,
        w_tilt: float = 10.0,
        w_angvel: float = 5.0,
        w_term_vel: float = 15.0,
        w_term_pos: float = 10.0,
        w_term_tilt: float = 10.0,
        glide_slope_ratio: float = 2.0,
        tilt_limit_high: float = 0.44,
        tilt_limit_low: float = 0.17,
        tilt_transition_alt: float = 15.0,
        angvel_limit: float = 5.0,
    ):
        super().__init__()
        self.w_ground = w_ground
        self.w_fuel = w_fuel
        self.w_glide = w_glide
        self.w_tilt = w_tilt
        self.w_angvel = w_angvel
        self.w_term_vel = w_term_vel
        self.w_term_pos = w_term_pos
        self.w_term_tilt = w_term_tilt
        self.glide_slope_ratio = glide_slope_ratio
        self.tilt_limit_high = tilt_limit_high
        self.tilt_limit_low = tilt_limit_low
        self.tilt_transition_alt = tilt_transition_alt
        self.angvel_limit = angvel_limit

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute (B,) constraint cost over (B, T, 17) trajectory."""
        pos = trajectory[:, :, 0:3]
        vel = trajectory[:, :, 3:6]
        quat = trajectory[:, :, 6:10]
        ang_vel = trajectory[:, :, 10:13]
        fuel = trajectory[:, :, 13:14]

        B, T, _ = trajectory.shape
        cost = torch.zeros(B, device=trajectory.device)

        for t in range(T):
            h_t = pos[:, t, 2]
            cost += self.w_ground * F.relu(-h_t) ** 2
            cost += self.w_fuel * F.relu(-fuel[:, t, 0]) ** 2

            lat_dist = torch.norm(pos[:, t, :2], dim=-1)
            cost += self.w_glide * F.relu(lat_dist - self.glide_slope_ratio * h_t - 5.0) ** 2

            tilt = tilt_from_quat_batch(quat[:, t])
            alpha = torch.clamp(h_t / self.tilt_transition_alt, 0.0, 1.0)
            tilt_limit = self.tilt_limit_low + (self.tilt_limit_high - self.tilt_limit_low) * alpha
            cost += self.w_tilt * F.relu(tilt - tilt_limit) ** 2

            angvel_mag = torch.norm(ang_vel[:, t], dim=-1)
            cost += self.w_angvel * F.relu(angvel_mag - self.angvel_limit) ** 2

        vel_T = vel[:, -1]
        pos_T = pos[:, -1]
        quat_T = quat[:, -1]
        cost += self.w_term_vel * torch.sum(vel_T ** 2, dim=-1)
        cost += self.w_term_pos * torch.sum(pos_T[:, :2] ** 2, dim=-1)
        cost += self.w_term_tilt * tilt_from_quat_batch(quat_T) ** 2

        return cost


def project_rocket_actions(actions: torch.Tensor, gimbal_limit: float = 0.0873) -> torch.Tensor:
    """Project (B, T, 7) action candidates to feasible actuator bounds."""
    a = actions.clone()
    a[..., 0:3] = a[..., 0:3].clamp(-1.0, 1.0)
    a[..., 3] = (a[..., 3] > 0.5).float()
    a[..., 4] = a[..., 4].clamp(0.0, 1.0)
    a[..., 4] = a[..., 4] * a[..., 3]
    a[..., 5:7] = a[..., 5:7].clamp(-gimbal_limit, gimbal_limit)
    return a


class PhysicsConstrainedCost:
    """Combined latent goal cost + physics constraint cost for MPC solvers."""

    def __init__(
        self,
        world_model,
        physics_model: RocketPhysicsModel | None = None,
        constraint_cost: RocketConstraintCost | None = None,
        lambda_constraint: float = 1.0,
        project_actions: bool = True,
        device: str = "cpu",
    ):
        self.wm = world_model
        self.physics = physics_model or RocketPhysicsModel()
        self.constraints = constraint_cost or RocketConstraintCost()
        self.lam = lambda_constraint
        self.project = project_actions
        self.device = device
        self.physics = self.physics.to(device)
        self.constraints = self.constraints.to(device)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute (B,) combined cost for action candidates."""
        if self.project:
            action_candidates = project_rocket_actions(action_candidates)

        goal_cost = self.wm.get_cost(info_dict, action_candidates)

        if self.lam > 0 and "proprio" in info_dict:
            proprio = info_dict["proprio"]
            if torch.is_tensor(proprio):
                state_0 = proprio.squeeze(1).to(self.device).float()
            else:
                state_0 = torch.from_numpy(proprio).squeeze(1).to(self.device).float()

            actions_phys = action_candidates.to(self.device).float()
            trajectory = self.physics.rollout(state_0, actions_phys)
            phys_cost = self.constraints(trajectory)
            goal_cost = goal_cost + self.lam * phys_cost.to(goal_cost.device)

        return goal_cost
