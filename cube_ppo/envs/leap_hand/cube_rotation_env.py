from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct
from mujoco import mjx

from cube_ppo import ROOT


@struct.dataclass
class CubeRotationConfig:
    """Config dataclass for the cube rotation task."""

    model_path: Union[Path, str] = ROOT + "/envs/leap_hand/scene.xml"

    # number of simulation steps for every control input
    physics_steps_per_control_step: int = 8

    # Reset noise scales
    joint_position_noise_scale: float = 0.5
    joint_velocity_noise_scale: float = 0.1

    # Observation noise scales
    stdev_obs: float = 0.005

    # Cost weights
    grasp_weight: float = 0.001
    position_weight: float = 100.0
    orientation_weight: float = 1.0

    # Distance from the target position above which we impose a position
    # penalty
    position_threshold: float = 0.015


class CubeRotationEnv(PipelineEnv):
    """Environment for training a cube rotation task.

    The goal is to get the cube to match a target orientation, while staying
    close to the middle of the hand.

    States: cube state (position and velocity) and hand state (pos and vel).
    Observations: full state of the cube and hand, and target orientation.
    Actions: joint position targets.
    """

    def __init__(self, config: Optional[CubeRotationConfig] = None) -> None:
        """Initialize the cube rotation environment."""
        if config is None:
            config = CubeRotationConfig()
        self.config = config
        mj_model = mujoco.MjModel.from_xml_path(config.model_path)
        sys = mjcf.load_model(mj_model)

        # Get sensor ids
        self.cube_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_position"
        )
        self.cube_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_orientation"
        )
        self.cube_linvel_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_linvel"
        )
        self.cube_angvel_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_angvel"
        )

        super().__init__(
            sys, n_frames=config.physics_steps_per_control_step, backend="mjx"
        )

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Resets the environment to an initial state."""
        # Set the hand to near the home position
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        q_hand = self.config.joint_position_noise_scale * jax.random.normal(
            pos_rng, (16,)
        )
        v_hand = self.config.joint_velocity_noise_scale * jax.random.normal(
            vel_rng, (16,)
        )

        # Set the cube to start just above the hand
        rng, p_rng, quat_rng = jax.random.split(rng, 3)
        start_pos = jnp.array([0.11, 0.0, 0.1]) + jax.random.uniform(
            p_rng, (3,), minval=-0.02, maxval=0.02
        )
        start_quat = self._sample_quaternion(quat_rng)
        q_cube = jnp.array([*start_pos, *start_quat])
        v_cube = jnp.zeros(6)

        # Set a random cube target orientation
        # https://stackoverflow.com/questions/31600717/
        rng, goal_rng = jax.random.split(rng)
        goal_quat = self._sample_quaternion(goal_rng)

        # Set the simulator state
        qpos = jnp.concatenate([q_hand, q_cube])
        qvel = jnp.concatenate([v_hand, v_cube])
        data = self.pipeline_init(qpos, qvel)
        data = data.tree_replace({"mocap_quat": goal_quat[None]})

        # Set other brax state fields (observation, reward, metrics, etc)
        obs = self._compute_obs(data, {})
        reward, done = jnp.zeros(2)
        metrics = {
            "reward": reward,
            "grasp_cost": 0.0,
            "position_err": 0.0,
            "orientation_err": 0.0,
            "goal_reached": 0.0,
        }
        info = {"rng": rng, "step": 0}  # TODO: include target in info
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        rng, rng_obs = jax.random.split(state.info["rng"])

        # Scale actions from [-1, 1] to joint limits
        u_min = self.sys.actuator_ctrlrange[:, 0]
        u_max = self.sys.actuator_ctrlrange[:, 1]
        action = (action + 1.0) / 2.0 * (u_max - u_min) + u_min

        # Apply the action
        data = self.pipeline_step(state.pipeline_state, action)

        # Compute the observation
        obs = self._compute_obs(data, state.info)
        obs += jax.random.normal(rng_obs, obs.shape) * self.config.stdev_obs

        # Calculate the reward
        reward, metrics = self._compute_reward(data, state.info)

        # Reset if the cube is dropped
        cube_z = data.qpos[18]
        done = jnp.where(cube_z < -0.1, 1.0, 0.0)
        reward -= done * 1000.0

        # Update the training state
        state.info["step"] += 1
        state.info["rng"] = rng
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _sample_quaternion(self, rng: jax.random.PRNGKey) -> jax.Array:
        """Sample a random quaternion."""
        # See https://stackoverflow.com/questions/31600717/
        u, v, w = jax.random.uniform(rng, (3,))
        return jnp.array(
            [
                jnp.sqrt(1 - u) * jnp.sin(2 * jnp.pi * v),
                jnp.sqrt(1 - u) * jnp.cos(2 * jnp.pi * v),
                jnp.sqrt(u) * jnp.sin(2 * jnp.pi * w),
                jnp.sqrt(u) * jnp.cos(2 * jnp.pi * w),
            ]
        )

    def _get_cube_position_err(self, data: mjx.Data) -> jax.Array:
        """Position of the cube relative to the target grasp position."""
        sensor_adr = self.sys.sensor_adr[self.cube_position_sensor]
        return data.sensordata[sensor_adr : sensor_adr + 3]

    def _get_cube_orientation_err(self, data: mjx.Data) -> jax.Array:
        """Orientation of the cube relative to the target grasp orientation."""
        sensor_adr = self.sys.sensor_adr[self.cube_orientation_sensor]
        cube_quat = data.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = data.mocap_quat[0]
        return mjx._src.math.quat_sub(cube_quat, goal_quat)

    def _get_cube_linvel(self, data: mjx.Data) -> jax.Array:
        """Linear velocity of the cube."""
        sensor_adr = self.sys.sensor_adr[self.cube_linvel_sensor]
        return data.sensordata[sensor_adr : sensor_adr + 3]

    def _get_cube_angvel(self, data: mjx.Data) -> jax.Array:
        """Angular velocity of the cube."""
        sensor_adr = self.sys.sensor_adr[self.cube_angvel_sensor]
        return data.sensordata[sensor_adr : sensor_adr + 3]

    def _compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jnp.ndarray:
        """Compute the observation from the simulator state."""
        # Hand joint positions and velocities
        q_hand = data.qpos[:16]
        v_hand = data.qvel[:16]

        # Cube position and orientation errors
        cube_pos_err = self._get_cube_position_err(data)
        cube_ori_err = self._get_cube_orientation_err(data)

        # Cube velocities (in the world frame)
        cube_linvel = self._get_cube_linvel(data)
        cube_angvel = self._get_cube_angvel(data)

        return jnp.concatenate(
            [
                q_hand,
                v_hand,
                cube_pos_err,
                cube_ori_err,
                cube_linvel,
                cube_angvel,
            ]
        )

    def _compute_reward(
        self, data: mjx.Data, info: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Compute the reward from the simulator state."""
        # Distance from a nominal grasp position
        grasp_cost = jnp.sum(jnp.square(data.ctrl))  # ctrl = target position

        cube_position = jnp.linalg.norm(self._get_cube_position_err(data))
        cube_orientation = jnp.linalg.norm(self._get_cube_orientation_err(data))
        goal_bonus = jnp.where(cube_orientation < 0.1, 250.0, 0.0)

        position_penalty = jnp.maximum(
            cube_position - self.config.position_threshold, 0.0
        )

        total_reward = (
            -self.config.orientation_weight * cube_orientation
            - self.config.position_weight * position_penalty
            - self.config.grasp_weight * grasp_cost
            + goal_bonus
        )

        metrics = {
            "reward": total_reward,
            "grasp_cost": grasp_cost,
            "position_err": cube_position,
            "orientation_err": cube_orientation,
            "goal_reached": jnp.float32(cube_orientation < 0.1),
        }

        return total_reward, metrics
