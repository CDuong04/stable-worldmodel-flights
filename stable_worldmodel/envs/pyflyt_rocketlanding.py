from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Literal

import numpy as np
import pybullet as p
import PyFlyt
from gymnasium.spaces import Box
from PyFlyt.gym_envs.rocket_envs.rocket_base_env import RocketBaseEnv

import stable_worldmodel as swm


class RocketLandingEnv(RocketBaseEnv):
    """Rocket Landing Environment.

    Actions: [finlet_x, finlet_y, finlet_roll, ignition, throttle, gimbal_x, gimbal_y]

    Observation (17D):
        [0:3] position, [3:6] velocity, [6:10] quaternion (wxyz),
        [10:13] angular_velocity, [13] fuel_fraction, [14:17] target_relative
    """

    def __init__(
        self,
        sparse_reward: bool = False,
        ceiling: float = 250.0,
        max_displacement: float = 200.0,
        max_duration_seconds: float = 30.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 40,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        super().__init__(
            start_pos=np.array([[0.0, 0.0, ceiling * 0.9]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            ceiling=ceiling,
            max_displacement=max_displacement,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64,
        )
        self.observation_space.low[13] = 0.0
        self.observation_space.high[13] = 1.0

        pyflyt_dir = os.path.dirname(os.path.realpath(PyFlyt.__file__))
        self.targ_obj_dir = os.path.join(pyflyt_dir, "models/landing_pad.urdf")

        self.sparse_reward = sparse_reward

        self.variation_space = swm.spaces.Dict(
            {
                "rocket": swm.spaces.Dict(
                    {
                        "body_color": swm.spaces.RGBBox(
                            init_value=np.array([255, 204, 0], dtype=np.uint8)
                        ),
                        "fin_color": swm.spaces.RGBBox(
                            init_value=np.array([51, 51, 51], dtype=np.uint8)
                        ),
                        "leg_color": swm.spaces.RGBBox(
                            init_value=np.array([0, 0, 0], dtype=np.uint8)
                        ),
                        "booster_color": swm.spaces.RGBBox(
                            init_value=np.array([51, 51, 51], dtype=np.uint8)
                        ),
                    }
                ),
                "pad": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array([200, 200, 200], dtype=np.uint8)
                        ),
                    }
                ),
                "environment": swm.spaces.Dict(
                    {
                        "sky_color": swm.spaces.RGBBox(
                            init_value=np.array([135, 206, 235], dtype=np.uint8)
                        ),
                        "start_height_ratio": swm.spaces.Box(
                            low=0.7, high=0.95, init_value=0.9,
                            shape=(), dtype=np.float32,
                        ),
                        "start_horizontal_offset": swm.spaces.Box(
                            low=-20.0, high=20.0,
                            init_value=np.array([0.0, 0.0], dtype=np.float32),
                            shape=(2,), dtype=np.float32,
                        ),
                        "start_tilt": swm.spaces.Box(
                            low=-0.2, high=0.2,
                            init_value=np.array([0.0, 0.0], dtype=np.float32),
                            shape=(2,), dtype=np.float32,
                        ),
                    }
                ),
            },
            sampling_order=["environment", "rocket", "pad"],
        )

        self.ceiling = ceiling
        self.original_start_pos = np.array([[0.0, 0.0, ceiling * 0.9]])
        self.modified_rocket_urdf = None
        self.modified_pad_urdf = None

    def _modify_rocket_urdf(self) -> str:
        pyflyt_dir = os.path.dirname(os.path.realpath(PyFlyt.__file__))
        original_urdf = os.path.join(pyflyt_dir, "models/vehicles/rocket/rocket.urdf")

        tree = ET.parse(original_urdf)
        root = tree.getroot()

        body_color = self.variation_space["rocket"]["body_color"].value / 255.0
        fin_color = self.variation_space["rocket"]["fin_color"].value / 255.0
        leg_color = self.variation_space["rocket"]["leg_color"].value / 255.0

        for material in root.findall(".//material[@name='yellow']"):
            color = material.find("color")
            color.set("rgba", f"{body_color[0]:.3f} {body_color[1]:.3f} {body_color[2]:.3f} 1.0")

        for material in root.findall(".//material[@name='grey']"):
            color = material.find("color")
            color.set("rgba", f"{fin_color[0]:.3f} {fin_color[1]:.3f} {fin_color[2]:.3f} 1.0")

        for material in root.findall(".//material[@name='black']"):
            color = material.find("color")
            color.set("rgba", f"{leg_color[0]:.3f} {leg_color[1]:.3f} {leg_color[2]:.3f} 1.0")

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tree.write(temp_file.name)
        temp_file.close()
        return temp_file.name

    def _modify_pad_urdf(self) -> str:
        tree = ET.parse(self.targ_obj_dir)
        root = tree.getroot()

        pad_color = self.variation_space["pad"]["color"].value / 255.0

        material = root.find(".//material[@name='pad_material']")
        if material is None:
            material = ET.Element("material", name="pad_material")
            color_elem = ET.SubElement(material, "color")
            color_elem.set("rgba", f"{pad_color[0]:.3f} {pad_color[1]:.3f} {pad_color[2]:.3f} 1.0")
            root.insert(0, material)
        else:
            color = material.find("color")
            color.set("rgba", f"{pad_color[0]:.3f} {pad_color[1]:.3f} {pad_color[2]:.3f} 1.0")

        visual = root.find(".//visual")
        if visual is not None:
            mat_ref = visual.find("material")
            if mat_ref is None:
                mat_ref = ET.SubElement(visual, "material")
            mat_ref.set("name", "pad_material")

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tree.write(temp_file.name)
        temp_file.close()
        return temp_file.name

    def reset(self, *, seed: None | int = None, options: None | dict[str, Any] = None) -> tuple[np.ndarray, dict]:
        if options is None:
            options = {"randomize_drop": False, "accelerate_drop": True}

        self.variation_space.seed(seed)
        self.variation_space.reset()

        variation_options = options.get("variation", [])
        if variation_options:
            from collections.abc import Sequence
            if not isinstance(variation_options, Sequence):
                raise ValueError("variation option must be a Sequence containing variation names to sample")
            self.variation_space.update(variation_options)

        start_height_ratio = self.variation_space["environment"]["start_height_ratio"].value
        start_offset = self.variation_space["environment"]["start_horizontal_offset"].value
        start_tilt = self.variation_space["environment"]["start_tilt"].value

        self.start_pos = np.array(
            [[start_offset[0], start_offset[1], self.ceiling * start_height_ratio]], dtype=np.float64
        )
        self.start_orn = np.array([[start_tilt[0], start_tilt[1], 0.0]], dtype=np.float64)

        starting_fuel_ratio = options.get('starting_fuel_ratio', 0.05) if options else 0.05

        super().begin_reset(
            seed=seed, options=options,
            drone_options={"starting_fuel_ratio": starting_fuel_ratio},
        )

        self.landing_pad_contact = 0.0
        self.ang_vel = np.zeros((3,))
        self.lin_vel = np.zeros((3,))
        self.lin_pos = np.zeros((3,))
        self.ground_lin_vel = np.zeros((3,))
        self.previous_ang_vel = np.zeros((3,))
        self.previous_lin_vel = np.zeros((3,))
        self.previous_lin_pos = np.zeros((3,))
        self.previous_ground_lin_vel = np.zeros((3,))

        if variation_options and (
            "all" in variation_options
            or any("rocket" in v for v in variation_options)
            or any("pad" in v for v in variation_options)
        ):
            if self.modified_pad_urdf:
                try:
                    os.unlink(self.modified_pad_urdf)
                except Exception:
                    pass
            self.modified_pad_urdf = self._modify_pad_urdf()
            pad_urdf_path = self.modified_pad_urdf
        else:
            pad_urdf_path = self.targ_obj_dir

        self.landing_pad_position = np.array([0.0, 0.0, 0.0])
        self.landing_pad_id = self.env.loadURDF(
            pad_urdf_path,
            basePosition=np.array([0.0, 0.0, 0.1]),
            useFixedBase=True,
        )

        super().end_reset(seed, options)

        if variation_options and ("all" in variation_options or any("rocket" in v for v in variation_options)):
            rocket_id = self.env.drones[0].Id
            body_color = self.variation_space["rocket"]["body_color"].value / 255.0
            fin_color = self.variation_space["rocket"]["fin_color"].value / 255.0
            leg_color = self.variation_space["rocket"]["leg_color"].value / 255.0
            booster_color = self.variation_space["rocket"]["booster_color"].value / 255.0

            num_joints = p.getNumJoints(rocket_id, physicsClientId=self.env._client)

            for i in range(-1, num_joints):
                if i == -1:
                    p.changeVisualShape(
                        rocket_id, i, rgbaColor=list(body_color) + [1.0], physicsClientId=self.env._client
                    )
                else:
                    joint_info = p.getJointInfo(rocket_id, i, physicsClientId=self.env._client)
                    link_name = joint_info[12].decode("utf-8")

                    if "fin" in link_name.lower():
                        p.changeVisualShape(
                            rocket_id, i, rgbaColor=list(fin_color) + [1.0], physicsClientId=self.env._client
                        )
                    elif "leg" in link_name.lower():
                        p.changeVisualShape(
                            rocket_id, i, rgbaColor=list(leg_color) + [1.0], physicsClientId=self.env._client
                        )
                    elif "booster" in link_name.lower():
                        p.changeVisualShape(
                            rocket_id, i, rgbaColor=list(booster_color) + [1.0], physicsClientId=self.env._client
                        )

        init_state_id = p.saveState(physicsClientId=self.env._client)
        p.resetBasePositionAndOrientation(
            self.env.drones[0].Id,
            [0.0, 0.0, 1.5],
            p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            physicsClientId=self.env._client,
        )
        self.current_goal = self.render()
        p.restoreState(stateId=init_state_id, physicsClientId=self.env._client)

        self.info["goal"] = self.current_goal
        # PyFlyt reuses self.info across steps; copy so wrapper mutations don't leak back.
        return self.state, dict(self.info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        state, reward, terminated, truncated, info = super().step(action)
        # PyFlyt reuses self.info across steps; copy so wrapper mutations don't leak back.
        info = dict(info)
        info["goal"] = self.current_goal
        return state, reward, terminated, truncated, info

    def compute_state(self) -> None:
        """Observation: [pos(3), vel(3), quat_wxyz(4), ang_vel(3), fuel(1), target_rel(3)]"""
        self.previous_ang_vel = self.ang_vel.copy()
        self.previous_lin_vel = self.lin_vel.copy()
        self.previous_lin_pos = self.lin_pos.copy()
        self.previous_ground_lin_vel = self.ground_lin_vel.copy()

        (
            self.ang_vel,
            self.ang_pos,
            self.lin_vel,
            self.lin_pos,
            quaternion,
        ) = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        self.ground_lin_vel = np.matmul(self.lin_vel, rotation.T)

        # aux_state: [lifting(4), booster(3: ignition, fuel_ratio, throttle), gimbal(2)]
        fuel_fraction = aux_state[5]
        target_relative = self.landing_pad_position - self.lin_pos

        # PyBullet quat is (xyzw), convert to (wxyz)
        quaternion_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        self.state = np.concatenate([
            self.lin_pos,
            self.lin_vel,
            quaternion_wxyz,
            self.ang_vel,
            np.array([fuel_fraction]),
            target_relative,
        ], axis=-1)

    def compute_term_trunc_reward(self) -> None:
        super().compute_base_term_trunc_reward(collision_ignore_mask=[self.env.drones[0].Id, self.landing_pad_id])

        if not self.sparse_reward:
            lateral_progress = float(
                np.linalg.norm(self.previous_lin_pos[:2]) - np.linalg.norm(self.lin_pos[:2])
            )
            vertical_progress = float(self.previous_lin_pos[-1] - self.lin_pos[-1])
            lateral_distance = np.linalg.norm(self.lin_pos[:2]) + 0.1

            deceleration_progress = (
                (self.ground_lin_vel[-1] - self.previous_ground_lin_vel[-1] + 1.0)
                / np.exp(self.lin_pos[-1])
                * (1.0 if (self.ground_lin_vel[-1] < 0.0) else -1.0)
            )

            self.reward += (
                -0.3
                + (0.3 / lateral_distance)
                + (10.0 * lateral_progress)
                + (0.2 * vertical_progress)
                + (4.0 * deceleration_progress)
                - (1.0 * abs(self.ang_vel[-1]))
                - (1.0 * np.linalg.norm(self.ang_pos[:2]))
            )

        if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
            self.landing_pad_contact = 1.0
            self.reward += 5.0 - (0.3 * abs(self.ground_lin_vel[-1]))
        else:
            self.landing_pad_contact = 0.0
            return

        # Fatal collision: ang_vel > 10 rad/s or world-frame speed > 5 m/s
        # Use ground_lin_vel (world frame) not lin_vel (body frame).
        # Body-frame velocity includes tilt-induced lateral components that
        # make even gentle touchdowns register as fatal when slightly tilted.
        if np.linalg.norm(self.ang_vel) > 10.0 or np.linalg.norm(self.ground_lin_vel) > 5.0:
            self.termination |= True
            self.info["fatal_collision"] = True
            return

        # Success: ang_vel < 5, world-frame speed < 2 m/s, tilt < 0.5 rad, lateral < 5m
        if (
            np.linalg.norm(self.ang_vel) < 5.0
            and np.linalg.norm(self.ground_lin_vel) < 2.0
            and np.linalg.norm(self.ang_pos[:2]) < 0.5
            and np.linalg.norm(self.lin_pos[:2]) < 5.0
        ):
            self.truncation |= True
            self.info["env_complete"] = True
            self.reward += 3.0
            return
