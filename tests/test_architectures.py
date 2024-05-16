import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from cube_ppo.architectures import (
    MLP,
    DiagonalQuadratic,
    Linear,
    LinearSystem,
    MLPWithPassthrough,
    Quadratic,
    print_module_summary,
)


def test_mlp_construction():
    """Create a simple MLP and verify sizes."""
    input_size = (3,)
    layer_sizes = (2, 3, 4)

    # Pseudo-random keys
    param_rng, tabulate_rng, input_rng = jax.random.split(
        jax.random.PRNGKey(0), 3
    )

    # Create the MLP
    mlp = MLP(layer_sizes=layer_sizes, bias=True)
    dummy_input = jnp.ones(input_size)
    params = mlp.init(param_rng, dummy_input)

    # Check the MLP's structure
    print_module_summary(mlp, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(input_rng, input_size)
    my_output = mlp.apply(params, my_input)
    assert my_output.shape[-1] == layer_sizes[-1]

    # Check number of parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    expected_num_params = 0
    sizes = input_size + layer_sizes
    for i in range(len(sizes) - 1):
        expected_num_params += sizes[i] * sizes[i + 1]  # weights
        expected_num_params += sizes[i + 1]  # biases
    assert num_params == expected_num_params


def test_mlp_save_load():
    """Verify that we can pickle an MLP."""
    rng = jax.random.PRNGKey(0)
    mlp = MLP(layer_sizes=(2, 3, 4))
    dummy_input = jnp.ones((3,))
    params = mlp.init(rng, dummy_input)

    original_output = mlp.apply(params, dummy_input)

    # Create a temporary path for saving stuff
    local_dir = Path("_test_mlp")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Save the MLP
    model_path = local_dir / "mlp.pkl"
    with Path(model_path).open("wb") as f:
        pickle.dump(mlp, f)

    # Save the parameters (weights)
    params_path = local_dir / "params.pkl"
    with Path(params_path).open("wb") as f:
        pickle.dump(params, f)

    # Load the MLP and parameters
    with Path(model_path).open("rb") as f:
        new_mlp = pickle.load(f)
    with Path(params_path).open("rb") as f:
        new_params = pickle.load(f)

    # Check that the loaded MLP gives the same output
    new_output = new_mlp.apply(new_params, dummy_input)
    assert jnp.allclose(original_output, new_output)

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_quadratic():
    """Test creating and evaluating a quadratic module."""
    input_size = (2, 5)

    # Create the quadratic module
    rng = jax.random.PRNGKey(0)
    net = Quadratic(input_size[-1])
    dummy_input = jnp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Print the network summary
    print_module_summary(net, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)

    assert my_output.shape[-1] == 1
    assert my_output.shape[0] == input_size[0]


def test_diagonal_quadratic():
    """Test creating and evaluating a quadratic module."""
    input_size = (2, 5)

    # Create the module
    rng = jax.random.PRNGKey(0)
    net = DiagonalQuadratic(input_size[-1])
    dummy_input = jnp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Print the network summary
    print_module_summary(net, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)

    assert my_output.shape[-1] == 1
    assert my_output.shape[0] == input_size[0]


def test_linear():
    """Test creating and evaluating a linear module."""
    input_size = (2, 5)
    output_size = 3

    # Create the linear module
    rng = jax.random.PRNGKey(0)
    net = Linear(output_size, bias=False)
    dummy_input = jnp.ones(input_size)
    params = net.init(rng, dummy_input)

    assert len(list(params["params"].keys())) == 1
    K = params["params"]["dense"]["kernel"]
    assert K.shape == (input_size[-1], output_size)

    # Print the network summary
    print_module_summary(net, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)

    assert my_output.shape[-1] == output_size
    assert my_output.shape[0] == input_size[0]

    manual_output = jnp.dot(my_input, K)
    assert jnp.allclose(my_output, manual_output)


def test_mlp_with_passthrough():
    """Test creating and evaluating an MLP with passthrough."""
    input_size = (2, 5)
    layer_sizes = (3, 4)

    # Create the MLP
    rng = jax.random.PRNGKey(0)
    net = MLPWithPassthrough(layer_sizes=layer_sizes, bias=True)
    dummy_input = jnp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Print the network summary
    print_module_summary(net, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)

    # Check sizes
    assert my_output.shape[-1] == layer_sizes[-1] + input_size[-1]
    assert my_output.shape[0] == input_size[0]

    # First part of the output should be the input
    assert jnp.allclose(my_output[:, : input_size[-1]], my_input)


def test_linear_system():
    """Test creating and evaluating a linear system."""
    rng = jax.random.PRNGKey(0)
    # Specify sizes
    nx = 5  # state dimension
    nu = 3  # control dimension
    extra_dims = (2, 4)  # batch/time/whatever else dimensions

    # Make the module
    init_rng, rng = jax.random.split(rng)
    net = LinearSystem(nx, nu)
    dummy_x = jnp.ones(extra_dims + (nx,))
    dummy_u = jnp.ones(extra_dims + (nu,))
    params = net.init(init_rng, dummy_x, dummy_u)

    # Forward pass through the network
    x_rng, u_rng, rng = jax.random.split(rng, 3)
    my_x = jax.random.normal(x_rng, extra_dims + (nx,))
    my_u = jax.random.normal(u_rng, extra_dims + (nu,))
    my_output = net.apply(params, my_x, my_u)

    # Check sizes
    assert my_output.shape == extra_dims + (nx,)

    # Check that the manual vector version makes sense
    x = my_x[0, 0, :]
    u = my_u[0, 0, :]
    A = params["params"]["A"]
    B = params["params"]["B"]

    x_next_manual = A @ x + B @ u
    assert jnp.allclose(my_output[0, 0, :], x_next_manual)


if __name__ == "__main__":
    test_mlp_construction()
    test_mlp_save_load()
    test_quadratic()
    test_diagonal_quadratic()
    test_linear()
    test_mlp_with_passthrough()
    test_linear_system()
