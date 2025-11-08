"""
Expert Policy wrapper for GNC Controller
Integrates the RocketLandingGNC controller with the stable-worldmodel-flights framework
"""

import numpy as np
from scipy.spatial.transform import Rotation
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
        
        if controller_params is None:
            controller_params = ControllerParams()
        
        self.gnc = RocketLandingGNC(params=controller_params, angle_representation="quaternion")
    
    def get_action(self, obs, goal_obs=None, **kwargs):
        if isinstance(obs, dict):
            if 'state' in obs:
                observation = obs['state']
            elif 'observation' in obs:
                observation = obs['observation']
            else:
                observation = obs
        else:
            observation = obs
        
        observation = np.asarray(observation, dtype=float).flatten()
        state_dict = parse_observation(observation, "quaternion")
        action = self.gnc.compute_control(state_dict)
        self.gnc.post_step_update()
        action = np.asarray(action, dtype=np.float32).flatten()
        
        return action

        
    def reset(self):
        self.gnc.reset()
    
    def get_telemetry(self):
        return self.gnc.get_telemetry()
    
    def get_fuel_report(self):
        return self.gnc.get_fuel_report()
    
    def set_env(self, env):
        super().set_env(env)
        
        if hasattr(env, 'observation_space'):
            obs_shape = env.observation_space.shape
            print(f"Environment observation space: {obs_shape}")
        
        if hasattr(env, 'action_space'):
            action_shape = env.action_space.shape
            print(f"Environment action space: {action_shape}")
