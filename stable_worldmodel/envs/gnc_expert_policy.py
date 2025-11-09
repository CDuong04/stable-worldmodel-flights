"""
Expert Policy wrapper for GNC Controller
Integrates the RocketLandingGNC controller with the stable-worldmodel-flights framework
"""

import numpy as np
from stable_worldmodel.policy import ExpertPolicy
from stable_worldmodel.envs.rocket_landing_gnc import RocketLandingGNC, ControllerParams


def parse_observation(observation: np.ndarray, angle_rep: str = "quaternion"):
    """Parse observation array into structured dictionary.
    obs[0:3]   - position (x, y, z) [meters]
    obs[3:6]   - velocity (vx, vy, vz) [m/s]
    obs[6:10]  - quaternion (w, x, y, z) [unitless]
    obs[10:13] - angular_velocity (wx, wy, wz) [rad/s]
    obs[13]    - fuel_fraction [0-1]
    obs[14:17] - target_relative (dx, dy, dz) [meters]
    """
    obs = np.asarray(observation, dtype=float).flatten()

    
    pos = obs[0:3]
    vel = obs[3:6]
    quat = obs[6:10]
    ang_vel = obs[10:13]
    fuel_obs = float(obs[13])
    target_rel = obs[14:17]
    
    return dict(
        position=pos,
        velocity=vel,
        quaternion=quat,
        angular_velocity=ang_vel,
        fuel_obs=fuel_obs,
        target_rel=target_rel
    )


class GNCExpertPolicy(ExpertPolicy):
    def __init__(self, controller_params=None, **kwargs):
        super().__init__(**kwargs)

        self.controller_params = controller_params or ControllerParams()
        self.controllers: list[RocketLandingGNC] = [self._make_controller()]

    def _make_controller(self) -> RocketLandingGNC:
        return RocketLandingGNC(params=self.controller_params, angle_representation="quaternion")

    def _ensure_controller_count(self, count: int) -> None:
        if count <= 0:
            raise ValueError("Controller count must be positive.")

        if len(self.controllers) < count:
            for _ in range(count - len(self.controllers)):
                self.controllers.append(self._make_controller())
        elif len(self.controllers) > count:
            self.controllers = self.controllers[:count]

    def _extract_observation_array(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            if "state" in obs:
                observation = obs["state"]
            elif "observation" in obs:
                observation = obs["observation"]
            else:
                observation = obs
        else:
            observation = obs
        return np.asarray(observation, dtype=float)

    def get_action(self, obs, goal_obs=None, **kwargs):
        observation = self._extract_observation_array(obs)
        batched = observation.ndim > 1
        obs_batch = observation if batched else observation[None, :]

        self._ensure_controller_count(obs_batch.shape[0])

        actions = []
        for idx, single_obs in enumerate(obs_batch):
            state_dict = parse_observation(single_obs, "quaternion")
            controller = self.controllers[idx]
            action = controller.compute_control(state_dict)
            controller.post_step_update()
            actions.append(np.asarray(action, dtype=np.float32).flatten())

        actions = np.stack(actions, axis=0)
        return actions if batched else actions[0]

    def reset(self):
        for controller in self.controllers:
            controller.reset()

    def get_telemetry(self):
        if len(self.controllers) == 1:
            return self.controllers[0].get_telemetry()
        return [controller.get_telemetry() for controller in self.controllers]

    def get_fuel_report(self):
        if len(self.controllers) == 1:
            return self.controllers[0].get_fuel_report()
        return [controller.get_fuel_report() for controller in self.controllers]

    def set_env(self, env):
        super().set_env(env)
        num_envs = getattr(env, "num_envs", 1)
        self._ensure_controller_count(num_envs)

        if hasattr(env, "observation_space"):
            obs_shape = env.observation_space.shape

        if hasattr(env, "action_space"):
            action_shape = env.action_space.shape
