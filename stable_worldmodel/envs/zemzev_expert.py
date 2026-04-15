"""ZEM/ZEV guidance controller for rocket landing."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from stable_worldmodel.policy import ExpertPolicy


@dataclass
class ZEMZEVParams:
    """Parameters for ZEM/ZEV guidance controller."""

    max_thrust: float = 7607.0
    min_frac: float = 0.20
    dry_mass: float = 138.2
    fuel_mass_max: float = 410.9
    g: float = 9.81
    C1: float = 6.0
    C2: float = 2.0
    kp_att: float = 5.0
    kd_att: float = 3.0
    gimbal_limit: float = 1.0
    t_go_min: float = 0.5
    cutoff_h: float = 0.3


def parse_obs(obs: np.ndarray) -> dict:
    """Parse observation vector into named components."""
    obs = np.asarray(obs, dtype=float).flatten()
    return {
        "position": obs[0:3].copy(),
        "velocity": obs[3:6].copy(),
        "quaternion": obs[6:10].copy(),
        "angular_velocity": obs[10:13].copy(),
        "fuel_frac": float(obs[13]),
        "target_rel": obs[14:17].copy(),
    }


def quat_to_rotmat(quat_wxyz):
    """Convert wxyz quaternion to rotation matrix."""
    xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return Rotation.from_quat(xyzw).as_matrix()


def quat_to_euler(quat_wxyz):
    """Convert wxyz quaternion to xyz Euler angles."""
    xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return Rotation.from_quat(xyzw).as_euler("xyz")


def body_to_world_vel(quat_wxyz, vel_body):
    """Transform velocity from body frame to world frame."""
    return quat_to_rotmat(quat_wxyz) @ vel_body


class ZEMZEVController:
    """ZEM/ZEV (Apollo E-guidance) controller for rocket landing."""

    def __init__(self, params: ZEMZEVParams = None):
        self.p = params or ZEMZEVParams()
        self.prev_gimbal = np.zeros(2)
        self._telem = {}

    def compute_control(self, state: dict) -> np.ndarray:
        """Compute control action from parsed state."""
        pos = state["position"]
        vel_body = state["velocity"]
        quat = state["quaternion"]
        ang_vel = state["angular_velocity"]
        fuel = state["fuel_frac"]
        euler = quat_to_euler(quat)

        vel_world = body_to_world_vel(quat, vel_body)
        h = pos[2]
        vz = vel_world[2]

        target_pos = pos + state["target_rel"]
        target_pos[2] = 0.0
        target_vel = np.zeros(3)

        mass = self.p.dry_mass + fuel * self.p.fuel_mass_max
        g_vec = np.array([0.0, 0.0, -self.p.g])

        lat_dist = np.linalg.norm(target_pos[:2] - pos[:2])

        if h < self.p.cutoff_h:
            gimbal_x, gimbal_y = self._pd_attitude(euler, ang_vel, 0.0, 0.0)
            finlets = self._compute_finlets(euler, ang_vel, vel_body)
            self._telem = {"phase": "cutoff", "h": round(h, 1), "lat": round(lat_dist, 1)}
            return np.array([finlets[0], finlets[1], finlets[2],
                             0.0, 0.0, gimbal_x, gimbal_y], dtype=np.float32)

        max_accel = self.p.max_thrust * 0.9 / mass
        max_decel = max_accel - self.p.g
        a_brake = max_decel / 1.4

        if h > 1.0 and a_brake > 0.1:
            fall_speed = max(-vz, 0.0)
            h_burn = (fall_speed**2 + 2.0 * self.p.g * h) / (2.0 * (a_brake + self.p.g))

            coast_dist = max(h - h_burn, 0.0)
            v_at_burn = np.sqrt(fall_speed**2 + 2.0 * self.p.g * coast_dist)

            t_coast = (v_at_burn - fall_speed) / self.p.g if coast_dist > 0.1 else 0.0
            t_brake = v_at_burn / a_brake if a_brake > 0.1 else 5.0
            t_go = max(t_coast + t_brake, self.p.t_go_min)
        else:
            t_go = self.p.t_go_min

        t_go = max(t_go, self.p.t_go_min)

        r_zem = (target_pos - pos) - vel_world * t_go - 0.5 * g_vec * t_go**2
        v_zev = (target_vel - vel_world) - g_vec * t_go

        a_cmd = self.p.C1 * r_zem / t_go**2 + self.p.C2 * v_zev / t_go

        F_des = a_cmd - g_vec
        F_des = a_cmd + np.array([0.0, 0.0, self.p.g])
        F_mag = np.linalg.norm(F_des)

        max_accel = self.p.max_thrust * 0.95 / mass
        min_accel = self.p.min_frac * self.p.max_thrust / mass

        desired_thrust = F_mag * mass
        desired_frac = desired_thrust / self.p.max_thrust
        desired_frac = np.clip(desired_frac, 0.0, 1.0)

        if desired_frac < self.p.min_frac * 0.5:
            ignition = 0.0
            pwm = 0.0
        else:
            desired_frac = max(desired_frac, self.p.min_frac)
            pwm = (desired_frac - self.p.min_frac) / (1.0 - self.p.min_frac)
            pwm = np.clip(pwm, 0.0, 1.0)
            ignition = 1.0

        if F_mag > 0.01:
            F_dir = F_des / F_mag
            target_pitch = np.arctan2(F_dir[0], F_dir[2])
            target_roll = np.arctan2(-F_dir[1], F_dir[2])
        else:
            target_pitch = 0.0
            target_roll = 0.0

        max_tilt = np.radians(35) if h > 10 else np.radians(20) if h > 3 else np.radians(10)
        target_pitch = np.clip(target_pitch, -max_tilt, max_tilt)
        target_roll = np.clip(target_roll, -max_tilt, max_tilt)

        gimbal_x, gimbal_y = self._pd_attitude(euler, ang_vel, target_roll, target_pitch)
        finlets = self._compute_finlets(euler, ang_vel, vel_body)

        phase = "burn" if ignition > 0.5 else "coast"
        self._telem = {
            "phase": phase, "h": round(h, 1), "vz_w": round(vz, 1),
            "t_go": round(t_go, 2), "lat": round(lat_dist, 1),
            "thr_frac": round(float(desired_frac), 3),
            "pwm": round(float(pwm), 3),
            "tilt_cmd": round(np.degrees(np.sqrt(target_pitch**2 + target_roll**2)), 1),
            "|ZEM|": round(float(np.linalg.norm(r_zem)), 1),
            "|ZEV|": round(float(np.linalg.norm(v_zev)), 1),
        }

        return np.array([finlets[0], finlets[1], finlets[2],
                         ignition, float(pwm),
                         gimbal_x, gimbal_y], dtype=np.float32)

    def _pd_attitude(self, euler, ang_vel, target_roll, target_pitch):
        kp = self.p.kp_att
        kd = self.p.kd_att
        gx_raw = kp * (euler[0] - target_roll) + kd * ang_vel[0]
        gy_raw = kp * (euler[1] - target_pitch) + kd * ang_vel[1]
        max_delta = 0.3
        gx = self.prev_gimbal[0] + np.clip(gx_raw - self.prev_gimbal[0], -max_delta, max_delta)
        gy = self.prev_gimbal[1] + np.clip(gy_raw - self.prev_gimbal[1], -max_delta, max_delta)
        gx = float(np.clip(gx, -self.p.gimbal_limit, self.p.gimbal_limit))
        gy = float(np.clip(gy, -self.p.gimbal_limit, self.p.gimbal_limit))
        self.prev_gimbal = np.array([gx, gy])
        return gx, gy

    def _compute_finlets(self, euler, ang_vel, vel_body):
        speed = np.linalg.norm(vel_body)
        gain = min(speed / 25.0, 1.0)
        kp = self.p.kp_att
        kd = self.p.kd_att
        fx = float(np.clip(-gain * (kp * euler[1] + kd * ang_vel[1]), -1.0, 1.0))
        fy = float(np.clip(-gain * (kp * euler[0] + kd * ang_vel[0]), -1.0, 1.0))
        fr = float(np.clip(-(kp * euler[2] + kd * ang_vel[2]), -1.0, 1.0))
        return (fx, fy, fr)

    def reset(self):
        """Reset controller state."""
        self.prev_gimbal = np.zeros(2)
        self._telem = {}

    def get_telemetry(self):
        """Return latest telemetry dict."""
        return self._telem


class ZEMZEVExpertPolicy(ExpertPolicy):
    """ZEM/ZEV expert policy compatible with stable_worldmodel.World."""

    def __init__(self, params: ZEMZEVParams = None, **kwargs):
        super().__init__(**kwargs)
        self.type = "zemzev_expert"
        self.params = params or ZEMZEVParams()
        self.controllers = [ZEMZEVController(self.params)]

    def _ensure_controllers(self, count):
        while len(self.controllers) < count:
            self.controllers.append(ZEMZEVController(self.params))
        if len(self.controllers) > count:
            self.controllers = self.controllers[:count]

    def get_action(self, obs, goal_obs=None, **kwargs):
        """Compute action for one or a batch of observations."""
        if isinstance(obs, dict):
            obs = obs.get("state", obs.get("observation", obs))
        obs = np.asarray(obs, dtype=float)
        batched = obs.ndim > 1
        obs_batch = obs if batched else obs[None, :]
        self._ensure_controllers(obs_batch.shape[0])
        actions = []
        for idx, single_obs in enumerate(obs_batch):
            state = parse_obs(single_obs)
            action = self.controllers[idx].compute_control(state)
            actions.append(action)
        actions = np.stack(actions)
        return actions if batched else actions[0]

    def reset(self):
        """Reset all controllers."""
        for c in self.controllers:
            c.reset()

    def get_telemetry(self):
        """Return telemetry from all controllers."""
        if len(self.controllers) == 1:
            return self.controllers[0].get_telemetry()
        return [c.get_telemetry() for c in self.controllers]
