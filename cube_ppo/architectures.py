from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def print_module_summary(module: nn.Module, input_shape: Sequence[int]):
    """Print a readable summary of a flax neural network module.

    Args:
        module: The flax module to summarize.
        input_shape: The shape of the input to the module.
    """
    # Create a dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones(input_shape)
    print(module.tabulate(rng, dummy_input, depth=1))


class MLP(nn.Module):
    """A simple pickle-able multi-layer perceptron.

    Args:
        layer_sizes: Sizes of all hidden layers and the output layer.
        activate_final: Whether to apply an activation function to the output.
        bias: Whether to use a bias in the linear layers.
    """

    layer_sizes: Sequence[int]
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        # TODO(vincekurtz): consider using jax control flow here. Note that
        # standard jax control flows (e.g. jax.lax.scan) do not play nicely with
        # flax, see for example https://github.com/google/flax/discussions/1283.
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                layer_size,
                use_bias=self.bias,
                kernel_init=nn.initializers.lecun_uniform(),
                name=f"dense_{i}",
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = nn.tanh(x)
        return x


class MLPWithPassthrough(nn.Module):
    """A simple multi-layer perceptron where the input is passed through.

        y = [x, MLP(x)]

    Args:
        layer_sizes: Sizes of all hidden layers and the MLP output layer.
        activate_final: Whether to apply activation to the MLP output.
        bias: Whether to use a bias in the linear layers.
    """

    layer_sizes: Sequence[int]
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        # Pass the input through the MLP
        mlp_output = MLP(
            layer_sizes=self.layer_sizes,
            activate_final=self.activate_final,
            bias=self.bias,
        )(x)

        # Concatenate the input and the MLP output
        return jnp.concatenate([x, mlp_output], axis=-1)


class Quadratic(nn.Module):
    """A module that computes a quadratic function of its input.

        y = x^T P x + b'x

    Note: the output is always a scalar.

    Args:
        input_size: Dimension of the input x
    """

    nx: int

    def setup(self):
        """Initialize the module."""
        self.P = self.param(
            "P", nn.initializers.lecun_uniform(), (self.nx, self.nx)
        )

    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        return jnp.einsum("ij,...i,...j->...", self.P, x, x)[..., jnp.newaxis]


class DiagonalQuadratic(nn.Module):
    """A module that computes a quadratic function of its input.

        y = x^T P x + b'x

    where P is diagonal.

    Note: the output is always a scalar.

    Args:
        input_size: Dimension of the input x
    """

    nx: int

    def setup(self):
        """Initialize the module."""
        self.P = self.param("P", nn.initializers.lecun_uniform(), (self.nx, 1))

    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        return jnp.einsum("i0,...i,...i->...", self.P, x, x)[..., jnp.newaxis]


class Linear(nn.Module):
    """A module that computes a linear function of its input.

            y = W x + b

    Args:
        output_size: Dimension of the output y
        bias: Whether to use the bias term b
    """

    output_size: int
    bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        return nn.Dense(
            self.output_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.lecun_uniform(),
            name="dense",
        )(x)


class LinearSystem(nn.Module):
    """A module that computes the output of a linear system.

        x_{t+1} = A x_t + B u_t

    Args:
        nx: Dimension of the state x
        nu: Dimension of the input u
    """

    nx: int
    nu: int

    def setup(self):
        """Initialize the module."""
        self.A = self.param(
            "A", nn.initializers.lecun_uniform(), (self.nx, self.nx)
        )
        self.B = self.param(
            "B", nn.initializers.lecun_uniform(), (self.nx, self.nu)
        )

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray):
        """Forward pass."""
        return jnp.matmul(x, self.A.T) + jnp.matmul(u, self.B.T)


class Identity(nn.Module):
    """A simple module that just passes the input to output, unchanged."""

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        return x
