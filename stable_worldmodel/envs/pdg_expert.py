"""Powered Descent Guidance (PDG) Expert Controller for Rocket Landing."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from stable_worldmodel.policy import ExpertPolicy


@dataclass
class PDGParams:
    """Parameters for PDG suicide-burn controller."""
    max_thrust: float = 7607.0
    min_frac: float = 0.20
    dry_mass: float = 138.2
    fuel_mass_max: float = 410.9
    g: float = 9.81
    kp_att: float = 5.0
    kd_att: float = 3.0
    gimbal_limit: float = 1.0
    safety_factor: float = 1.4
    cutoff_h: float = 0.3
    thrust_margin: float = 0.92
    kp_z: float = 3.0
    kp_nav: float = 2.5
    kp_lat_vel: float = 2.5
    max_coast_tilt: float = 0.35
    pre_aim_gain: float = 0.8


def parse_obs(obs: np.ndarray) -> dict:
    """Parse flat observation vector into named components."""
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
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return Rotation.from_quat(xyzw).as_matrix()


def quat_to_euler(quat_wxyz):
    """Convert wxyz quaternion to xyz Euler angles."""
    xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return Rotation.from_quat(xyzw).as_euler("xyz")


def body_to_world_vel(quat_wxyz, vel_body):
    """Transform body-frame velocity to world frame."""
    R = quat_to_rotmat(quat_wxyz)
    return R @ vel_body


class PDGController:
    """Suicide-burn (hoverslam) powered descent guidance controller."""

    def __init__(self, params: PDGParams = None):
        self.p = params or PDGParams()
        self.prev_gimbal = np.zeros(2)
        self._telem = {}

    def compute_control(self, state: dict) -> np.ndarray:
        """Compute 7-dim control vector from parsed state."""
        pos = state["position"]
        vel_body = state["velocity"]
        quat = state["quaternion"]
        ang_vel = state["angular_velocity"]
        fuel = state["fuel_frac"]
        euler = quat_to_euler(quat)

        vel_world = body_to_world_vel(quat, vel_body)
        h = pos[2]
        vz = vel_world[2]
        vx, vy = vel_world[0], vel_world[1]

        target_pos = pos + state["target_rel"]
        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        lat_dist = np.sqrt(dx**2 + dy**2)

        mass = self.p.dry_mass + fuel * self.p.fuel_mass_max
        max_thrust_eff = self.p.max_thrust * self.p.thrust_margin
        max_accel = max_thrust_eff / mass
        max_decel = max_accel - self.p.g
        tilt_mag = np.sqrt(euler[0]**2 + euler[1]**2)

        fall_speed = max(-vz, 0.0)
        a_brake = max_decel / self.p.safety_factor if max_decel > 0.1 else 0.1
        h_burn = (fall_speed**2 + 2.0 * self.p.g * h) / (2.0 * (a_brake + self.p.g))
        h_burn += self.p.cutoff_h

        if vz < -1.0:
            t_go = max(h / (-vz), 0.3)
        elif h > 1.0:
            t_go = max(np.sqrt(2.0 * h / self.p.g), 0.5)
        else:
            t_go = 0.5

        if h < self.p.cutoff_h:
            gimbal_x, gimbal_y = self._pd_attitude(euler, ang_vel, 0.0, 0.0)
            finlets = self._compute_finlets(euler, ang_vel, vel_body)
            self._telem = {"phase": "cutoff", "h": round(h, 1),
                           "vz_w": round(vz, 1), "lat": round(lat_dist, 1)}
            return np.array([finlets[0], finlets[1], finlets[2],
                             0.0, 0.0, gimbal_x, gimbal_y], dtype=np.float32)

        if h > h_burn and vz < -0.5:
            v_lat_need_x = dx / max(t_go, 1.0)
            v_lat_need_y = dy / max(t_go, 1.0)
            max_lat_v = np.clip(h * 0.1, 2.0, 20.0)
            v_lat_mag = np.sqrt(v_lat_need_x**2 + v_lat_need_y**2)
            if v_lat_mag > max_lat_v:
                s = max_lat_v / v_lat_mag
                v_lat_need_x *= s
                v_lat_need_y *= s

            needs_lateral = lat_dist > 2.0 or np.sqrt((vx - v_lat_need_x)**2 +
                                                        (vy - v_lat_need_y)**2) > 2.0

            if needs_lateral and h > 30.0:
                ax_cmd_coast = self.p.kp_lat_vel * (v_lat_need_x - vx)
                ay_cmd_coast = self.p.kp_lat_vel * (v_lat_need_y - vy)

                Fz_coast = self.p.g * 0.7

                F_coast = np.array([ax_cmd_coast, ay_cmd_coast, Fz_coast])
                F_coast_mag = np.linalg.norm(F_coast)

                desired_thrust_coast = F_coast_mag * mass
                desired_frac_coast = desired_thrust_coast / self.p.max_thrust
                desired_frac_coast = np.clip(desired_frac_coast, self.p.min_frac, 0.5)
                pwm_coast = (desired_frac_coast - self.p.min_frac) / (1.0 - self.p.min_frac)
                pwm_coast = np.clip(pwm_coast, 0.0, 1.0)

                if F_coast_mag > 0.01:
                    F_dir = F_coast / F_coast_mag
                    target_pitch = np.arctan2(F_dir[0], F_dir[2])
                    target_roll = np.arctan2(-F_dir[1], F_dir[2])
                else:
                    target_pitch, target_roll = 0.0, 0.0

                target_pitch = np.clip(target_pitch, -self.p.max_coast_tilt, self.p.max_coast_tilt)
                target_roll = np.clip(target_roll, -self.p.max_coast_tilt, self.p.max_coast_tilt)

                gimbal_x, gimbal_y = self._pd_attitude(euler, ang_vel, target_roll, target_pitch)
                finlets = self._compute_finlets(euler, ang_vel, vel_body)

                self._telem = {
                    "phase": "divert", "h": round(h, 1), "vz_w": round(vz, 1),
                    "h_burn": round(h_burn, 1), "lat": round(lat_dist, 1),
                    "thr_frac": round(float(desired_frac_coast), 3),
                    "tilt_cmd": round(np.degrees(np.sqrt(target_pitch**2 + target_roll**2)), 1),
                }
                return np.array([finlets[0], finlets[1], finlets[2],
                                 1.0, float(pwm_coast), gimbal_x, gimbal_y], dtype=np.float32)

            else:
                target_pitch = np.arctan2(v_lat_need_x - vx, max(fall_speed, 5.0))
                target_roll = np.arctan2(-(v_lat_need_y - vy), max(fall_speed, 5.0))
                target_pitch *= self.p.pre_aim_gain
                target_roll *= self.p.pre_aim_gain
                target_pitch = np.clip(target_pitch, -self.p.max_coast_tilt, self.p.max_coast_tilt)
                target_roll = np.clip(target_roll, -self.p.max_coast_tilt, self.p.max_coast_tilt)

                gimbal_x, gimbal_y = self._pd_attitude(euler, ang_vel, target_roll, target_pitch)
                finlets = self._compute_finlets(euler, ang_vel, vel_body)

                self._telem = {
                    "phase": "coast", "h": round(h, 1), "vz_w": round(vz, 1),
                    "h_burn": round(h_burn, 1), "lat": round(lat_dist, 1),
                    "tilt_cmd": round(np.degrees(np.sqrt(target_pitch**2 + target_roll**2)), 1),
                }
                return np.array([finlets[0], finlets[1], finlets[2],
                                 0.0, 0.0, gimbal_x, gimbal_y], dtype=np.float32)

        a_brake = max_decel / self.p.safety_factor
        h_eff = max(h - self.p.cutoff_h, 0.01)
        vz_ref = -np.sqrt(2.0 * a_brake * h_eff)
        vz_ref = max(vz_ref, -100.0)

        speed_err = vz - vz_ref

        az_cmd = a_brake - self.p.kp_z * speed_err
        az_cmd = np.clip(az_cmd, 0.0, max_decel * 1.3)

        k_nav = np.clip(self.p.kp_nav / max(t_go, 0.3), 0.3, 5.0)
        vx_des = k_nav * dx
        vy_des = k_nav * dy

        max_lat_v = np.clip(h * 0.3, 1.0, 20.0)
        lat_v_cmd = np.sqrt(vx_des**2 + vy_des**2)
        if lat_v_cmd > max_lat_v:
            s = max_lat_v / lat_v_cmd
            vx_des *= s
            vy_des *= s

        if h < 5.0 and lat_dist < 3.0:
            vx_des *= 0.2
            vy_des *= 0.2

        ax_cmd = self.p.kp_lat_vel * (vx_des - vx)
        ay_cmd = self.p.kp_lat_vel * (vy_des - vy)

        Fz = az_cmd + self.p.g

        lat_urgency = min(lat_dist / 5.0, 1.0)
        if h < 3.0:
            tilt_lim = np.radians(5 + 15 * lat_urgency)
        elif h < 10.0:
            tilt_lim = np.radians(12 + 18 * lat_urgency)
        elif h < 30.0:
            tilt_lim = np.radians(20 + 15 * lat_urgency)
        elif h < 100.0:
            tilt_lim = np.radians(30 + 5 * lat_urgency)
        else:
            tilt_lim = np.radians(35)

        max_lat_accel = max(Fz, 0.1) * np.tan(tilt_lim)
        lat_accel_mag = np.sqrt(ax_cmd**2 + ay_cmd**2)
        if lat_accel_mag > max_lat_accel:
            s = max_lat_accel / lat_accel_mag
            ax_cmd *= s
            ay_cmd *= s

        F_des = np.array([ax_cmd, ay_cmd, Fz])
        F_mag = np.linalg.norm(F_des)

        desired_thrust = F_mag * mass
        desired_frac = desired_thrust / self.p.max_thrust
        desired_frac = np.clip(desired_frac, self.p.min_frac, 1.0)

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

        if h < 3.0:
            max_tilt = np.radians(5 + 15 * lat_urgency)
        elif h < 10.0:
            max_tilt = np.radians(15 + 15 * lat_urgency)
        elif h < 30.0:
            max_tilt = np.radians(25 + 10 * lat_urgency)
        else:
            max_tilt = np.radians(35)

        target_pitch = np.clip(target_pitch, -max_tilt, max_tilt)
        target_roll = np.clip(target_roll, -max_tilt, max_tilt)

        gimbal_x, gimbal_y = self._pd_attitude(euler, ang_vel, target_roll, target_pitch)
        finlets = self._compute_finlets(euler, ang_vel, vel_body)

        self._telem = {
            "phase": "burn", "h": round(h, 1), "vz_w": round(vz, 1),
            "vz_ref": round(vz_ref, 1), "lat": round(lat_dist, 1),
            "thr_frac": round(float(desired_frac), 3),
            "pwm": round(float(pwm), 3),
            "tilt_cmd": round(np.degrees(np.sqrt(target_pitch**2 + target_roll**2)), 1),
            "h_burn": round(h_burn, 1),
        }

        return np.array([finlets[0], finlets[1], finlets[2],
                         ignition, float(pwm),
                         gimbal_x, gimbal_y], dtype=np.float32)

    def _pd_attitude(self, euler, ang_vel, target_roll, target_pitch):
        """PD attitude controller returning gimbal commands."""
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
        """Compute aerodynamic finlet deflections for attitude stabilization."""
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
        """Return latest telemetry dictionary."""
        return self._telem


class PDGExpertPolicy(ExpertPolicy):
    """PDG Expert Policy compatible with stable_worldmodel.World."""

    def __init__(self, params: PDGParams = None, **kwargs):
        super().__init__(**kwargs)
        self.type = "pdg_expert"
        self.params = params or PDGParams()
        self.controllers = [PDGController(self.params)]

    def _ensure_controllers(self, count):
        """Ensure we have exactly `count` controllers."""
        while len(self.controllers) < count:
            self.controllers.append(PDGController(self.params))
        if len(self.controllers) > count:
            self.controllers = self.controllers[:count]

    def get_action(self, obs, goal_obs=None, **kwargs):
        """Compute expert action for one or a batch of observations."""
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
