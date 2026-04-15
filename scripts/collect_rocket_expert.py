"""Collect expert rocket landing data with images for world model training."""

import argparse
from pathlib import Path

import stable_worldmodel.envs.rocket_landing_ext
import stable_worldmodel as swm
from stable_worldmodel.envs.rocket_landing_gnc import RocketLandingGNC, ControllerParams
from stable_worldmodel.envs.pdg_expert import parse_obs
from stable_worldmodel.policy import ExpertPolicy

import numpy as np


class GNCExpertPolicy(ExpertPolicy):
    """GNC Convex MPC expert wrapped for the World class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "gnc_expert"
        self.controllers = []

    def _ensure_controllers(self, count):
        while len(self.controllers) < count:
            self.controllers.append(RocketLandingGNC())
        if len(self.controllers) > count:
            self.controllers = self.controllers[:count]

    def get_action(self, obs, goal_obs=None, **kwargs):
        """Compute action from 17D state observation."""
        if isinstance(obs, dict):
            obs_data = obs.get("state", obs.get("observation", None))
            if obs_data is None:
                for v in obs.values():
                    if isinstance(v, np.ndarray):
                        obs_data = v
                        break
        else:
            obs_data = obs

        obs_data = np.asarray(obs_data, dtype=float)
        batched = obs_data.ndim > 1
        obs_batch = obs_data if batched else obs_data[None, :]
        self._ensure_controllers(obs_batch.shape[0])

        actions = []
        for idx, single_obs in enumerate(obs_batch):
            state = parse_obs(single_obs)
            obs_dict = {
                "position": state["position"],
                "velocity": state["velocity"],
                "quaternion": state["quaternion"],
                "angular_velocity": state["angular_velocity"],
                "fuel_obs": state["fuel_frac"],
                "target_rel": state["target_rel"],
            }
            action = self.controllers[idx].compute_control(obs_dict)
            self.controllers[idx].post_step_update()
            actions.append(action)

        actions = np.stack(actions)
        return actions if batched else actions[0]

    def reset(self):
        for c in self.controllers:
            c.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dataset-name", type=str, default="rocket_expert_default")
    args = parser.parse_args()

    print(f"Collecting {args.episodes} episodes with GNC expert")
    print(f"Dataset: {args.dataset_name}")

    world = swm.World(
        "swm/PFRocketLanding-v0",
        num_envs=args.num_envs,
        image_shape=(args.image_size, args.image_size),
        max_episode_steps=1200,
        render_mode="rgb_array",
    )

    policy = GNCExpertPolicy()
    world.set_policy(policy)

    world.record_dataset(
        args.dataset_name,
        episodes=args.episodes,
        seed=args.seed,
    )

    print("Done.")


if __name__ == "__main__":
    main()
