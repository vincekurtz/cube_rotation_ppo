from pathlib import Path
from typing import Any, Dict, Optional, Union

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
    physics_steps_per_control_step: int = 1

    # Reset noise scales
    joint_position_noise_scale: float = 0.1
    joint_velocity_noise_scale: float = 0.1

    # Observation noise scales
    stdev_obs: float = 0.0

    # Cost weights
    cube_position_weight: float = 0.0
    cube_orientation_weight: float = 0.0
    cube_velocity_weight: float = 0.0
    grasp_weight: float = 1.0
    joint_velocity_weight: float = 0.0
    acutation_weight: float = 0.0


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

        self.q_grasp = jnp.array(mj_model.keyframe("home").qpos)
        self.v_grasp = jnp.zeros_like(self.q_grasp)

        super().__init__(
            sys, n_frames=config.physics_steps_per_control_step, backend="mjx"
        )

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Resets the environment to an initial state."""
        # Set the hand to near the home position
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        q_hand = (
            self.q_grasp
            + self.config.joint_position_noise_scale
            * jax.random.normal(pos_rng, (16,))
        )
        v_hand = (
            self.v_grasp
            + self.config.joint_velocity_noise_scale
            * jax.random.normal(vel_rng, (16,))
        )

        # Set a random cube state just above the hand
        # TODO

        # Set a random cube target
        # TODO

        # Set the simulator state
        qpos = q_hand  # TODO: add the cube state
        qvel = v_hand
        data = self.pipeline_init(qpos, qvel)

        # Set other brax state fields (observation, reward, metrics, etc)
        obs = self._compute_obs(data, {})
        reward, done = jnp.zeros(2)
        metrics = {"reward": reward}
        info = {"rng": rng, "step": 0}  # TODO: include target in info
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        rng, rng_obs = jax.random.split(state.info["rng"])

        # Apply the action
        data = self.pipeline_step(state.pipeline_state, action)

        # Compute the observation
        obs = self._compute_obs(data, state.info)
        obs += jax.random.normal(rng_obs, obs.shape) * self.config.stdev_obs

        # Calculate the reward
        reward = self._compute_reward(data, state.info)

        # Reset if the cube is dropped
        # TODO

        # Update the training state
        state.info["step"] += 1
        state.info["rng"] = rng
        state.metrics["reward"] = reward
        return state.replace(pipeline_state=data, obs=obs, reward=reward)

    def _compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jnp.ndarray:
        """Compute the observation from the simulator state."""
        # TODO: include the target orientation.
        q = data.qpos
        v = data.qvel
        return jnp.concatenate([q, v])

    def _compute_reward(
        self, data: mjx.Data, info: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute the reward from the simulator state."""
        q = data.qpos
        v = data.qvel

        grasp_cost = jnp.sum(jnp.square(q - self.q_grasp))
        joint_vel_cost = jnp.sum(jnp.square(v))

        total_reward = (
            -self.config.grasp_weight * grasp_cost
            - self.config.joint_velocity_weight * joint_vel_cost
        )
        return total_reward
