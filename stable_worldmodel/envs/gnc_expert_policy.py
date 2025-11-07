"""
Expert Policy wrapper for GNC Controller
Integrates the RocketLandingGNC controller with the stable-worldmodel-flights framework
"""

import numpy as np
from scipy.spatial.transform import Rotation
from policy import ExpertPolicy
from rocket_landing_gnc import RocketLandingGNC, ControllerParams


def parse_observation(observation: np.ndarray, angle_rep: str = "quaternion"):
    obs = np.asarray(observation, dtype=float)
    pos = obs[0:3] if obs.size >= 3 else np.zeros(3)
    vel = obs[3:6] if obs.size >= 6 else np.zeros(3)
    idx = 6
    if angle_rep == "quaternion":
        quat = obs[idx:idx+4] if obs.size >= idx+4 else np.array([1, 0, 0, 0], dtype=float)
        idx += 4
    else:
        eul = obs[idx:idx+3] if obs.size >= idx+3 else np.zeros(3)
        q = Rotation.from_euler("xyz", eul).as_quat()
        quat = np.array([q[3], q[0], q[1], q[2]], dtype=float)
        idx += 3
    ang_vel = obs[idx:idx+3] if obs.size >= idx+3 else np.zeros(3)
    idx += 3
    fuel_obs = None
    if obs.size > idx:
        potential_fuel = float(obs[idx])
        if 0.0 <= potential_fuel <= 1.0:
            fuel_obs = potential_fuel
        idx += 1
    target_rel = obs[idx:idx+3] if obs.size >= idx+3 else np.zeros(3)
    return dict(position=pos, velocity=vel, quaternion=quat,
                angular_velocity=ang_vel, fuel_obs=fuel_obs, target_rel=target_rel)


class GNCExpertPolicy(ExpertPolicy):
    def __init__(self, controller_params=None, **kwargs):
        super().__init__(**kwargs)
        
        # Create the GNC controller with provided or default parameters
        if controller_params is None:
            controller_params = ControllerParams()
        
        self.gnc = RocketLandingGNC(params=controller_params, angle_representation="quaternion")
    
    def get_action(self, info_dict, **kwargs):

        state_dict = parse_observation(info_dict['state'], "quaternion")
        
        action = self.gnc.compute_control(state_dict)
        
        self.gnc.post_step_update()
        
        return action
    
    def reset(self):
        self.gnc.reset()
    
    def get_telemetry(self):
        return self.gnc.get_telemetry()
    
    def get_fuel_report(self):
        return self.gnc.get_fuel_report()
    
    def set_env(self, env):
        super().set_env(env)
        
        obs_shape = env.observation_space.shape
        
        action_shape = env.action_space.shape