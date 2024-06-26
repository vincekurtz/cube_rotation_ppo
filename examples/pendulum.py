import pickle
import sys
import time

import jax
import mujoco
import mujoco.viewer
import numpy as np
from brax import envs
from brax.training.distribution import NormalTanhDistribution
from mujoco import mjx

from cube_ppo.architectures import MLP
from cube_ppo.envs.pendulum.swingup_env import PendulumSwingupEnv
from cube_ppo.ppo_utils import (
    BraxPPONetworksWrapper,
    make_policy_function,
    train_ppo,
)

"""
Use PPO to train a swingup policy for a simple inverted pendulum
"""


def train():
    """Train the swingup policy and save it to a file."""
    # Create policy and value functions
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(64, 64, 2)),  # action + log probability
        value_network=MLP(layer_sizes=(64, 64, 1)),
        action_distribution=NormalTanhDistribution,
    )

    # Train the policy
    _, params = train_ppo(
        env=PendulumSwingupEnv,
        network_wrapper=network_wrapper,
        save_path="/tmp/pendulum_ppo.pkl",
        num_timesteps=50_000_000,
        num_evals=10,
        reward_scaling=0.1,
        episode_length=200,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0,
        num_envs=1024,
        batch_size=512,
        seed=0,
    )


def test():
    """Test the swingup policy with an interactive mujoco simulation."""
    # Create a brax environment
    envs.register_environment("pendulum_swingup", PendulumSwingupEnv)
    env = envs.get_environment("pendulum_swingup")

    # Extract the mujoco system model
    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Load the trained policy
    with open("/tmp/pendulum_ppo.pkl", "rb") as f:
        network_and_params = pickle.load(f)
    network_wrapper = network_and_params["network_wrapper"]
    params = network_and_params["params"]

    # Create a policy function
    policy = make_policy_function(
        network_wrapper=network_wrapper,
        params=params,
        observation_size=3,
        action_size=1,
        normalize_observations=True,
    )
    jit_policy = jax.jit(policy)

    # Start an interactive simulation
    rng = jax.random.PRNGKey(0)
    dt = float(env.dt)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Use the fixed camera
        viewer.cam.fixedcamid = 0
        viewer.cam.type = 2

        while viewer.is_running():
            start_time = time.time()
            act_rng, rng = jax.random.split(rng)

            # Get an observation from the environment
            obs = env._compute_obs(mjx.put_data(mj_model, mj_data), {})

            # Get an action from the policy
            act, _ = jit_policy(obs, act_rng)
            act = env.config.u_max * act

            # Apply the policy and step the simulation
            mj_data.ctrl[:] = np.array(act)
            for _ in range(env._n_frames):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    usage_message = "Usage: python pendulum.py [train|test]"

    if len(sys.argv) != 2:
        print(usage_message)
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    else:
        print(usage_message)
        sys.exit(1)
