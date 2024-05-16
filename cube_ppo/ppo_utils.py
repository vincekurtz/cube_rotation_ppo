import pickle
from datetime import datetime

import flax.linen as nn
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.base import PipelineEnv
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo.networks import PPONetworks, make_inference_fn
from brax.training.types import Params
from flax import struct
from tensorboardX import SummaryWriter

"""
Helper utilities for interfacing with Brax's Proximal Policy Optimization (PPO)
implementation.
"""


@struct.dataclass
class BraxPPONetworksWrapper:
    """A lightweight wrapper around brax's PPONetworks.

    Allows us to more easily save and load networks with non-default architectures.
    """

    policy_network: nn.Module
    value_network: nn.Module
    action_distribution: distribution.ParametricDistribution

    def make_ppo_networks(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        check_sizes: bool = True,
    ) -> PPONetworks:
        """Create a PPONetworks object, compatible with brax's ppo.train() function.

        Args:
            observation_size: Size of the input (observation).
            action_size: Size of the policy output (action).
            preprocess_observations_fn: Function to preprocess (e.g. normalize) observations.
            check_sizes: Whether to check that the output sizes of the policy and value networks match the action and value distributions.

        Returns:
            A PPONetworks object.
        """
        # Create an action distribution. The policy network should output the
        # parameters of this distribution.
        action_dist = self.action_distribution(event_size=action_size)

        # Set up a dummy observation for parameter initialization.
        dummy_observation = jnp.zeros((1, observation_size))

        if check_sizes:
            rng = jax.random.PRNGKey(0)

            # Check that the output size of the policy network matches the size of
            # the action distribution.
            dummy_params = self.policy_network.init(rng, dummy_observation)
            policy_output = self.policy_network.apply(
                dummy_params, dummy_observation
            )
            assert (
                policy_output.shape[-1] == action_dist.param_size
            ), f"policy network output size {policy_output.shape[-1]} does not match action distribution size {action_dist.param_size}"

            # Check that the output size of the value network is 1.
            dummy_value_params = self.value_network.init(rng, dummy_observation)
            value_output = self.value_network.apply(
                dummy_value_params, dummy_observation
            )
            assert (
                value_output.shape[-1] == 1
            ), f"value network output size {value_output.shape} does not match expected size 1"

        # Create the policy network, a FeedForwardNetwork that contains an "init"
        # and an "apply" function.
        def policy_init(key):
            """Initialize the policy network from a random key."""
            return self.policy_network.init(key, dummy_observation)

        def policy_apply(processor_params, policy_params, obs):
            """Apply the policy given the parameters and an observation."""
            obs = preprocess_observations_fn(obs, processor_params)
            return self.policy_network.apply(policy_params, obs)

        # Create the value network. This is just like the policy network, but with a 1D output.
        def value_init(key):
            """Initialize the value network from a random key."""
            return self.value_network.init(key, dummy_observation)

        def value_apply(processor_params, value_params, obs):
            """Apply the value function given the parameters and an observation."""
            obs = preprocess_observations_fn(obs, processor_params)
            return jnp.squeeze(
                self.value_network.apply(value_params, obs), axis=-1
            )

        return PPONetworks(
            policy_network=networks.FeedForwardNetwork(
                init=policy_init, apply=policy_apply
            ),
            value_network=networks.FeedForwardNetwork(
                init=value_init, apply=value_apply
            ),
            parametric_action_distribution=action_dist,
        )


def train_ppo(
    env: PipelineEnv,
    network_wrapper: BraxPPONetworksWrapper,
    save_path: str = None,
    tensorboard_logdir: str = None,
    **kwargs,
):
    """Train a PPO agent and save the trained policy and parameters to disk.

    Args:
        env: The MJX environment to train in.
        network_wrapper: A BraxPPONetworksWrapper object containing the policy and value networks.
        save_path: The path to save the trained policy and parameters to.
        tensorboard_logdir: The directory to save tensorboard logs to.
        **kwargs: Additional arguments to pass to brax's ppo.train() function.

    Returns:
        make_inference_fn: A function that takes parameters and returns a policy function.
        params: The trained parameters.
    """
    # Initialize the environment
    print("Initializing environment...")
    envs.register_environment("ppo_training_env", env)
    env = envs.get_environment("ppo_training_env")

    # A separate eval env is required if domain randomization is used
    eval_env = envs.get_environment("ppo_training_env")

    # Define a tensorboard logging callback
    if tensorboard_logdir is not None:
        logdir = tensorboard_logdir
    else:
        logdir = f"/tmp/tops_tensorboard/ppo_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Setting up TensorBoard logging in {logdir}...")

    writer = SummaryWriter(logdir)
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Logs progress during RL."""
        print(f"  Steps: {num_steps}, Reward: {metrics['eval/episode_reward']}")
        times.append(datetime.now())

        # Write all metrics to tensorboard
        for key, val in metrics.items():
            if isinstance(val, jax.Array):
                val = float(val)  # we need floats for logging
            writer.add_scalar(key, val, num_steps)

    # Train the PPO agent
    print("Training...")
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        network_factory=network_wrapper.make_ppo_networks,
        eval_env=eval_env,
        **kwargs,
    )

    print(f"Time to jit: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    # Save the trained policy and parameters to disk
    if save_path is not None:
        print(f"Saving policy and parameters to {save_path}...")
        network_and_params = {
            "network_wrapper": network_wrapper,
            "params": params,
        }
        with open(save_path, "wb") as f:
            pickle.dump(network_and_params, f)

    print("Done!")
    return make_inference_fn, params


def make_policy_function(
    network_wrapper: BraxPPONetworksWrapper,
    params: Params,
    observation_size: int,
    action_size: int,
    normalize_observations: bool = True,
    deterministic: bool = True,
):
    """Create a policy function from a trained network.

    Args:
        network_wrapper: A BraxPPONetworksWrapper object containing the policy and value network structures.
        params: The trained parameters.
        observation_size: The size of the observation space.
        action_size: The size of the action space.
        normalize_observations: Whether to normalize observations (e.g. policy was trained with normalize_observations=True).
        deterministic: Whether to use a deterministic policy.

    Returns:
        policy: A function that takes an observation and a random key, and
          returns an action and some extra data (e.g. log probability)

    Note: the policy function is not jitted, in most cases you will want to jit
    this function before using it in a performance-critical context.
    """
    if normalize_observations:
        preprocess_observations_fn = running_statistics.normalize
    else:
        preprocess_observations_fn = types.identity_observation_preprocessor

    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    make_policy = make_inference_fn(ppo_networks)
    policy = make_policy(params, deterministic=deterministic)
    return policy
