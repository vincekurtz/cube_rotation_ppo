import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from brax.training import distribution
from brax.training.agents.ppo.networks import PPONetworks
from brax.training.types import Params

from cube_ppo.architectures import MLP
from cube_ppo.envs.pendulum.swingup_env import PendulumSwingupEnv
from cube_ppo.ppo_utils import (
    BraxPPONetworksWrapper,
    make_policy_function,
    train_ppo,
)


def test_ppo_wrapper():
    """Test the BraxPPONetworksWrapper."""
    observation_size = 3
    action_size = 2

    with pytest.raises(AssertionError):
        # We should get an error if the policy network's output doesn't match
        # the size of the action distribution (mean + variance)
        network_wrapper = BraxPPONetworksWrapper(
            policy_network=MLP(layer_sizes=(512, 2)),
            value_network=MLP(layer_sizes=(512, 1)),
            action_distribution=distribution.NormalTanhDistribution,
        )
        network_wrapper.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
        )

    with pytest.raises(AssertionError):
        # We should get an error if the value network's output isn't 1D
        network_wrapper = BraxPPONetworksWrapper(
            policy_network=MLP(layer_sizes=(512, 4)),
            value_network=MLP(layer_sizes=(512, 2)),
            action_distribution=distribution.NormalTanhDistribution,
        )
        network_wrapper.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
        )

    # We should end up with a PPONetworks object if everything is correct
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(512, 4)),
        value_network=MLP(layer_sizes=(512, 1)),
        action_distribution=distribution.NormalTanhDistribution,
    )
    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
    )
    assert isinstance(ppo_networks, PPONetworks)


def test_ppo_networks_io():
    """Test saving and loading PPONetworks."""
    # Create a PPONetworks object
    observation_size = 3
    action_size = 2
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(512, 4)),
        value_network=MLP(layer_sizes=(512, 1)),
        action_distribution=distribution.NormalTanhDistribution,
    )
    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
    )
    assert isinstance(ppo_networks, PPONetworks)

    # Save to a file
    local_dir = Path("_test_ppo_networks_io")
    local_dir.mkdir(parents=True, exist_ok=True)
    model_path = local_dir / "ppo_networks.pkl"
    with Path(model_path).open("wb") as f:
        pickle.dump(network_wrapper, f)

    # Load from a file and check that the network is the same
    with Path(model_path).open("rb") as f:
        new_network_wrapper = pickle.load(f)
    new_ppo_networks = new_network_wrapper.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
    )
    assert isinstance(new_ppo_networks, PPONetworks)
    assert jax.tree_util.tree_structure(
        ppo_networks
    ) == jax.tree_util.tree_structure(new_ppo_networks)

    # Clean up
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_train_ppo():
    """Test train wrapper for a simple PPO agent."""
    # Set up a random key
    rng = jax.random.PRNGKey(0)

    # Create a policy and value functions for a pendulum swingup task
    observation_size = 3
    action_size = 1
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(16, 2)),
        value_network=MLP(layer_sizes=(16, 1)),
        action_distribution=distribution.NormalTanhDistribution,
    )

    # Set up a temporary directory for saving the policy
    local_dir = Path("_test_train_ppo")
    local_dir.mkdir(parents=True, exist_ok=True)
    save_path = local_dir / "pendulum_policy.pkl"

    # Train the agent
    make_policy, params = train_ppo(
        env=PendulumSwingupEnv,
        network_wrapper=network_wrapper,
        save_path=save_path,
        tensorboard_logdir=local_dir,
        num_timesteps=1000,
        num_evals=3,
        reward_scaling=0.1,
        episode_length=200,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=8,
        num_updates_per_batch=2,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0,
        num_envs=64,
        batch_size=32,
        seed=0,
    )

    # Run a forward pass through the trained policy
    policy = make_policy(params, deterministic=True)

    # Check that the policy returns the correct action size
    obs_rng, act_rng = jax.random.split(rng)
    obs = jax.random.normal(obs_rng, (observation_size,))
    action, _ = policy(obs, act_rng)
    assert action.shape == (action_size,)

    # Load the trained policy from disk
    with Path(save_path).open("rb") as f:
        loaded_network_and_params = pickle.load(f)
    loaded_network_wrapper = loaded_network_and_params["network_wrapper"]
    loaded_params = loaded_network_and_params["params"]

    assert isinstance(loaded_network_wrapper, BraxPPONetworksWrapper)

    # Check that the loaded policy returns the same action
    loaded_policy = make_policy_function(
        loaded_network_wrapper,
        loaded_params,
        observation_size,
        action_size,
        normalize_observations=True,
        deterministic=True,
    )
    new_action, _ = loaded_policy(obs, act_rng)
    assert jnp.allclose(action, new_action)

    # Clean up
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_ppo_wrapper()
    test_ppo_networks_io()
    test_train_ppo()
