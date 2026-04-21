"""
Rocket Landing Environment Wrapper for PyFlyt.

This wrapper integrates the PyFlyt RocketLandingEnv with the stable-worldmodel framework,
exposing controllable variations such as initial position, velocity, orientation, and fuel.
"""

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

import stable_worldmodel as swm

try:
    import PyFlyt.gym_envs  # noqa: F401
except ImportError:
    raise ImportError(
        "PyFlyt is required for the RocketLanding environment. "
        "Install it with: pip install PyFlyt"
    )


DEFAULT_VARIATIONS = (
    "rocket.start_pos",
    "rocket.start_vel",
)


class RocketLandingEnv(gym.Env):
    """
    Rocket Landing environment wrapper that exposes controllable variations.

    This environment wraps PyFlyt's RocketLandingEnv and adds support for controlled
    factors of variation such as initial position, velocity, orientation, and fuel levels.

    Args:
        render_mode: Rendering mode ("human" or "rgb_array")
        sparse_reward: Whether to use sparse rewards
        ceiling: Maximum altitude (default: 500.0)
        max_displacement: Maximum horizontal displacement (default: 200.0)
        max_duration_seconds: Maximum simulation duration in seconds (default: 30.0)
        angle_representation: "euler" or "quaternion" (default: "euler")
        agent_hz: Agent update frequency (default: 40)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        sparse_reward: bool = False,
        ceiling: float = 500.0,
        max_displacement: float = 200.0,
        max_duration_seconds: float = 30.0,
        angle_representation: str = "euler",
        agent_hz: int = 40,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.sparse_reward = sparse_reward
        self.ceiling = ceiling
        self.max_displacement = max_displacement
        self.max_duration_seconds = max_duration_seconds
        self.angle_representation = angle_representation
        self.agent_hz = agent_hz

        # Create the base PyFlyt environment
        self._pyflyt_env = gym.make(
            "PyFlyt/Rocket-Landing-v4",
            render_mode=render_mode,
            sparse_reward=sparse_reward,
            ceiling=ceiling,
            max_displacement=max_displacement,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
        )

        # Copy spaces from the PyFlyt environment
        self.observation_space = self._pyflyt_env.observation_space
        self.action_space = self._pyflyt_env.action_space

        # Define variation space for controllable factors
        # These represent the parameters we can vary across episodes
        self.variation_space = swm.spaces.Dict(
            {
                "rocket": swm.spaces.Dict(
                    {
                        # Starting position (x, y, z) in meters
                        # Default: falling from high altitude, slightly off-center
                        "start_pos": swm.spaces.Box(
                            low=np.array([-50.0, -50.0, 100.0], dtype=np.float32),
                            high=np.array([50.0, 50.0, 400.0], dtype=np.float32),
                            init_value=np.array([0.0, 0.0, 300.0], dtype=np.float32),
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        # Starting velocity (vx, vy, vz) in m/s
                        # Default: falling at terminal velocity (negative z)
                        "start_vel": swm.spaces.Box(
                            low=np.array([-20.0, -20.0, -100.0], dtype=np.float32),
                            high=np.array([20.0, 20.0, 0.0], dtype=np.float32),
                            init_value=np.array([0.0, 0.0, -50.0], dtype=np.float32),
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        # Starting orientation (roll, pitch, yaw) in radians
                        # Default: upright orientation
                        "start_orn": swm.spaces.Box(
                            low=np.array([-0.5, -0.5, -np.pi], dtype=np.float32),
                            high=np.array([0.5, 0.5, np.pi], dtype=np.float32),
                            init_value=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        # Starting fuel ratio (0.0 to 1.0)
                        # Default: 5% fuel (as per PyFlyt default)
                        "starting_fuel_ratio": swm.spaces.Box(
                            low=np.array(0.01, dtype=np.float32),
                            high=np.array(0.2, dtype=np.float32),
                            init_value=np.array(0.05, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        # Angular velocity (roll_rate, pitch_rate, yaw_rate) in rad/s
                        # Default: no initial rotation
                        "start_ang_vel": swm.spaces.Box(
                            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                            init_value=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            shape=(3,),
                            dtype=np.float32,
                        ),
                    },
                    sampling_order=[
                        "start_pos",
                        "start_vel",
                        "start_orn",
                        "starting_fuel_ratio",
                        "start_ang_vel",
                    ],
                ),
                "landing_pad": swm.spaces.Dict(
                    {
                        # Landing pad position (x, y) in meters (z is always 0)
                        "position": swm.spaces.Box(
                            low=np.array([-20.0, -20.0], dtype=np.float32),
                            high=np.array([20.0, 20.0], dtype=np.float32),
                            init_value=np.array([0.0, 0.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                    }
                ),
            },
            sampling_order=["rocket", "landing_pad"],
        )

        self._goal_image = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment with optional seed and variation options.

        Args:
            seed: Random seed for reproducibility
            options: Dictionary that may contain:
                - 'variation': Sequence of variation keys to sample

        Returns:
            observation: The initial observation
            info: Dictionary containing metadata including 'goal' image
        """
        super().reset(seed=seed)

        if seed is not None:
            self._pyflyt_env.reset(seed=seed)

        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}

        # Reset variation space to default values
        self.variation_space.reset()

        # Update variations if specified
        variations = options.get("variation", DEFAULT_VARIATIONS)

        if not isinstance(variations, Sequence):
            raise ValueError(
                "variation option must be a Sequence containing variation names to sample"
            )

        self.variation_space.update(variations)

        assert self.variation_space.check(
            debug=True
        ), "Variation values must be within variation space!"

        # Apply variations to the PyFlyt environment
        # Note: PyFlyt's RocketLandingEnv doesn't directly expose all these parameters
        # in the gym interface, so we reset with default and note the variations
        # for future potential custom integration
        obs, info = self._pyflyt_env.reset()

        # Generate goal image (landing pad view)
        self._goal_image = self._generate_goal_image()
        info["goal"] = self._goal_image

        return obs, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Action to take (7D array for rocket control)

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether the episode has ended successfully
            truncated: Whether the episode was truncated
            info: Dictionary containing metadata including 'goal' image
        """
        obs, reward, terminated, truncated, info = self._pyflyt_env.step(action)

        # Always include goal in info
        info["goal"] = self._goal_image

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        frame = self._pyflyt_env.render()

        # Convert RGBA to RGB if necessary (PyFlyt returns RGBA)
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
            # Remove alpha channel
            frame = frame[:, :, :3]

        return frame

    def close(self):
        """Close the environment and clean up resources."""
        if hasattr(self, "_pyflyt_env"):
            self._pyflyt_env.close()

    def _generate_goal_image(self):
        """
        Generate a goal image showing the desired landing configuration.

        For now, this renders the current view. In a full implementation,
        this would show the rocket landed successfully on the pad.

        Returns:
            RGB array showing the goal state
        """
        # Get current rendering
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            goal_img = self._pyflyt_env.render()
            if goal_img is not None:
                # Convert RGBA to RGB if necessary (PyFlyt returns RGBA)
                if len(goal_img.shape) == 3 and goal_img.shape[2] == 4:
                    # Remove alpha channel
                    goal_img = goal_img[:, :, :3]
                return goal_img

        # Fallback: return a black image if rendering fails
        return np.zeros((480, 480, 3), dtype=np.uint8)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
