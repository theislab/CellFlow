from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from scaleflow._types import Layers_t
from scaleflow.networks import _utils as nn_utils

__all__ = ["PhenotypePredictor"]


class PhenotypePredictor(nn.Module):
    """Predicts scalar phenotype from condition embedding.

    Parameters
    ----------
    hidden_dims
        Hidden layer dimensions for the MLP.
    dropout_rate
        Dropout rate.
    act_fn
        Activation function.
    output_dim
        Output dimension (typically 1 for scalar phenotype).
    """

    hidden_dims: Sequence[int] = (256, 128, 64)
    dropout_rate: float = 0.0
    act_fn: Any = nn.silu
    output_dim: int = 1

    def setup(self):
        """Initialize the modules."""
        layers = [(nn.Dense, {"features": dim}) for dim in self.hidden_dims]
        self.hidden_layers = nn_utils._get_layers(layers, dropout_rate=self.dropout_rate)
        self.output_layer = nn.Dense(self.output_dim)

    @nn.compact
    def __call__(self, condition_embedding: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass.

        Parameters
        ----------
        condition_embedding
            Condition embedding of shape ``(batch_size, embedding_dim)``.
        training
            Whether the model is in training mode.

        Returns
        -------
        Phenotype prediction of shape ``(batch_size, output_dim)``.
        """
        out = condition_embedding
        for layer in self.hidden_layers:
            out = layer(out)
            if training and self.dropout_rate > 0:
                out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not training)
            out = self.act_fn(out)
        return self.output_layer(out)

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        condition_embedding_dim: int,
    ) -> train_state.TrainState:
        """Create the training state.

        Parameters
        ----------
        rng
            Random key.
        optimizer
            Optimizer.
        condition_embedding_dim
            Dimension of the condition embedding.

        Returns
        -------
        Training state.
        """
        condition_embedding = jnp.ones((1, condition_embedding_dim))
        params = self.init(rng, condition_embedding, training=False)["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

