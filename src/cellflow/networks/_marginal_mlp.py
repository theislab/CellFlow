from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state


class Block(nn.Module):
    """
    Block of a neural network.

    Parameters
    ----------
    dim
        Input dimension.
    out_dim
        Output dimension.
    num_layers
        Number of layers.
    act_fn
        Activation function.
    """

    dim: int = 128
    out_dim: int = 32
    num_layers: int = 3
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(self.dim, name=f"fc{i}")(x)
            x = self.act_fn(x)
        return nn.Dense(self.out_dim, name="fc_final")(x)


class MLPMarginal(nn.Module):
    """
    Neural network parameterizing a reweighting function.

    Parameters
    ----------
    hidden_dim
        Hidden dimension.
    num_layers
        Number of layers.
    act_fn
        Activation function.
    """

    hidden_dim: int
    num_layers: int = 3
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @property
    def is_potential(self) -> bool:
        return True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x
        z = Block(dim=self.hidden_dim, out_dim=1, num_layers=self.num_layers, act_fn=self.act_fn)(z)
        return nn.softplus(z)

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
    ) -> train_state.TrainState:
        params = self.init(rng, jnp.ones((1, input_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
