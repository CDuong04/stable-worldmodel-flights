from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from stable_worldmodel.policy import ExpertPolicy


@dataclass
class SuicideBurnParams:
    max_thrust: float = 7607.0
    min_thrust_frac: float = 0.39
    gimbal_limit: float = 0.262  # 15 degrees
    dry_mass: float = 159.0
    fuel_mass_max: float = 410.9
    g: float = 9.81
    target_landing_speed: float = 0.5
    safety_margin: float = 2.0
    kp_att: float = 15.0
    kd_att: float = 12.0


class SuicideBurnController:
    """Three-phase suicide burn controller: COAST -> BURN -> FINAL."""

    def __init__(self, params: SuicideBurnParams = None):
        self.params = params or SuicideBurnParams()
        self.phase = "COAST"
        self.burn_start_altitude = None
        self.prev_gimbal = np.zeros(2)

    def compute_control(self, state: dict) -> np.ndarray:
        pos = state['position']
        vel = state['velocity']
        quat = state['quaternion']
        ang_vel = state['angular_velocity']
        fuel_frac = state.get('fuel_obs', 0.05)

        h = pos[2]
        vz = vel[2]

        mass = self.params.dry_mass + fuel_frac * self.params.fuel_mass_max
        max_thrust_accel = self.params.max_thrust / mass
        max_decel = max_thrust_accel - self.params.g

        h_stop = (vz ** 2) / (2 * max_decel) if vz < 0 else 0
        commit_altitude = h_stop * self.params.safety_margin + 5.0

        if self.phase == "COAST":
            if h <= commit_altitude or (h < 10 and vz < -5):
                self.phase = "BURN"
                self.burn_start_altitude = h

        if self.phase == "BURN":
            if (h < 20 and abs(vz) < 5) or (h < 30 and abs(vz) < 2):
                self.phase = "FINAL"

        gimbal_x, gimbal_y, finlets = self._compute_attitude_control(quat, ang_vel, vel, pos)

        if self.phase == "COAST":
            throttle = 0.0
            ignition = 0.0
        elif self.phase == "BURN":
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            rot = Rotation.from_quat(quat_xyzw)
            euler = rot.as_euler('xyz')
            tilt_mag = np.sqrt(euler[0]**2 + euler[1]**2)

            target_vz_final = -3.0

            if vz < target_vz_final:
                v_sq_diff = vz**2 - target_vz_final**2
                h_brake = v_sq_diff / (2 * max_decel) * 1.2
            else:
                h_brake = 0

            if vz > 2:
                throttle = 0.0
            elif vz > 0:
                throttle = 0.0
            elif vz > target_vz_final:
                throttle = 0.0
            elif h < h_brake + 3:
                excess_speed = (-vz) - (-target_vz_final)
                throttle = min(1.0, 0.4 + 0.1 * excess_speed)
            elif vz > -10:
                throttle = 0.5 if h < h_brake + 15 else 0.0
            elif vz > -18:
                throttle = 0.7 if h < h_brake + 20 else 0.0
            else:
                throttle = 1.0

            if tilt_mag > np.radians(20):
                throttle = 0.0
            elif tilt_mag > np.radians(15) and throttle < 0.3:
                throttle = 0.3

            ignition = 1.0 if throttle > 0 else 0.0
        else:  # FINAL
            if h > 8:
                target_vz = -2.5
            elif h > 4:
                target_vz = -1.2
            elif h > 1.5:
                target_vz = -0.6
            else:
                target_vz = -0.4

            if vz < target_vz:
                v_sq_diff = vz**2 - target_vz**2
                h_stop = v_sq_diff / (2 * max_decel) * 1.1
            else:
                h_stop = 0

            if vz > 0.3:
                throttle = 0.0
                ignition = 0.0
            elif vz > -0.2:
                throttle = 0.0
                ignition = 0.0
            elif h < h_stop + 0.5:
                throttle = 0.9
                ignition = 1.0
            elif h < h_stop + 2:
                throttle = 0.6
                ignition = 1.0
            elif vz < target_vz - 1.5:
                throttle = 0.5
                ignition = 1.0
            elif vz < target_vz - 0.3:
                throttle = 0.3
                ignition = 1.0
            else:
                throttle = 0.0
                ignition = 0.0

        return np.array([
            finlets[0], finlets[1], finlets[2],
            ignition, throttle, gimbal_x, gimbal_y,
        ], dtype=np.float32)

    def _compute_attitude_control(self, quat, ang_vel, vel, pos):
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
        rot = Rotation.from_quat(quat_xyzw)
        euler = rot.as_euler('xyz')

        kp_att = self.params.kp_att
        kd_att = self.params.kd_att

        px, py = pos[0], pos[1]
        vx, vy = vel[0], vel[1]
        h = pos[2]
        vz = vel[2]

        tilt_magnitude = np.sqrt(euler[0]**2 + euler[1]**2)

        t_go = max(h / max(-vz, 1.0), 1.0) if vz < -0.5 else max(h / 5.0, 2.0)
        alt_scale = min(h / 25.0, 1.0) if h < 25 else 1.0

        if tilt_magnitude < np.radians(5):
            tilt_scale = 1.0
        elif tilt_magnitude < np.radians(8):
            tilt_scale = 0.5
        elif tilt_magnitude < np.radians(12):
            tilt_scale = 0.2
        elif tilt_magnitude < np.radians(18):
            tilt_scale = 0.1
        else:
            tilt_scale = 0.05

        kp_lateral = 0.06 * min(t_go / 3.0, 3.0) * alt_scale * tilt_scale
        kd_lateral = 0.26 * alt_scale * max(tilt_scale, 0.2)

        ang_vel_magnitude = np.sqrt(ang_vel[0]**2 + ang_vel[1]**2)
        if ang_vel_magnitude > 0.5:
            spin_scale = max(0.1, 1.0 - (ang_vel_magnitude - 0.5) * 2)
        else:
            spin_scale = 1.0

        kp_lateral *= spin_scale
        kd_lateral *= spin_scale

        target_pitch = -kp_lateral * px - kd_lateral * vx
        target_roll = kp_lateral * py + kd_lateral * vy

        lateral_dist = np.sqrt(px**2 + py**2)
        lateral_vel = np.sqrt(vx**2 + vy**2)

        if h < 8:
            max_tilt = np.radians(3)
        elif h < 15:
            max_tilt = np.radians(6)
        elif h < 30:
            max_tilt = np.radians(10)
        elif lateral_dist > 25 or lateral_vel > 3:
            max_tilt = np.radians(15)
        elif lateral_dist > 15:
            max_tilt = np.radians(12)
        elif lateral_dist > 8:
            max_tilt = np.radians(10)
        else:
            max_tilt = np.radians(7)

        target_pitch = np.clip(target_pitch, -max_tilt, max_tilt)
        target_roll = np.clip(target_roll, -max_tilt, max_tilt)

        gimbal_x_raw = kp_att * (euler[0] - target_roll) + kd_att * ang_vel[0]
        gimbal_y_raw = kp_att * (euler[1] - target_pitch) + kd_att * ang_vel[1]

        dt = 0.025  # 40 Hz
        max_delta = 4.0 * dt

        gimbal_x = self.prev_gimbal[0] + np.clip(gimbal_x_raw - self.prev_gimbal[0], -max_delta, max_delta)
        gimbal_y = self.prev_gimbal[1] + np.clip(gimbal_y_raw - self.prev_gimbal[1], -max_delta, max_delta)

        gimbal_x = np.clip(gimbal_x, -self.params.gimbal_limit, self.params.gimbal_limit)
        gimbal_y = np.clip(gimbal_y, -self.params.gimbal_limit, self.params.gimbal_limit)

        self.prev_gimbal = np.array([gimbal_x, gimbal_y])

        speed = np.linalg.norm(vel)
        finlet_gain = min(speed / 30.0, 1.0)

        finlet_x = np.clip(-finlet_gain * (kp_att * euler[1] + kd_att * ang_vel[1]), -1.0, 1.0)
        finlet_y = np.clip(-finlet_gain * (kp_att * euler[0] + kd_att * ang_vel[0]), -1.0, 1.0)
        finlet_roll = np.clip(-kp_att * euler[2] - kd_att * ang_vel[2], -1.0, 1.0)

        return gimbal_x, gimbal_y, (finlet_x, finlet_y, finlet_roll)

    def reset(self):
        self.phase = "COAST"
        self.burn_start_altitude = None
        self.prev_gimbal = np.zeros(2)

    def get_telemetry(self):
        return {'phase': self.phase, 'burn_start_altitude': self.burn_start_altitude}


def parse_observation(observation: np.ndarray) -> dict:
    obs = np.asarray(observation, dtype=float).flatten()
    return {
        'position': obs[0:3].copy(),
        'velocity': obs[3:6].copy(),
        'quaternion': obs[6:10].copy(),
        'angular_velocity': obs[10:13].copy(),
        'fuel_obs': float(obs[13]),
        'target_rel': obs[14:17].copy(),
    }


class SuicideBurnPolicy(ExpertPolicy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.controllers = [SuicideBurnController()]

    def _ensure_controller_count(self, count):
        while len(self.controllers) < count:
            self.controllers.append(SuicideBurnController())
        if len(self.controllers) > count:
            self.controllers = self.controllers[:count]

    def get_action(self, obs, goal_obs=None, **kwargs):
        if isinstance(obs, dict):
            obs = obs.get('state', obs.get('observation', obs))
        obs = np.asarray(obs, dtype=float)
        batched = obs.ndim > 1
        obs_batch = obs if batched else obs[None, :]

        self._ensure_controller_count(obs_batch.shape[0])

        actions = []
        for idx, single_obs in enumerate(obs_batch):
            state = parse_observation(single_obs)
            action = self.controllers[idx].compute_control(state)
            actions.append(action)

        actions = np.stack(actions)
        return actions if batched else actions[0]

    def reset(self):
        for c in self.controllers:
            c.reset()

    def get_telemetry(self):
        if len(self.controllers) == 1:
            return self.controllers[0].get_telemetry()
        return [c.get_telemetry() for c in self.controllers]
