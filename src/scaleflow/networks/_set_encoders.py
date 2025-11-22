from collections.abc import Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from flax.typing import FrozenDict

from scaleflow._types import ArrayLike, Layers_separate_input_t, Layers_t
from scaleflow.networks import _utils as nn_utils

__all__ = [
    "ConditionEncoder",
]


class ConditionEncoder(nn_utils.BaseModule):
    """
    Encoder for conditions represented as sets of perturbations.

    Parameters
    ----------
    output_dim
        Dimensionality of the output.
    condition_mode
        Mode of the encoder, should be one of:

        - ``'deterministic'``: Learns condition encoding point-wise.
        - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.
    regularization
        Regularization strength in the latent space:

        - For deterministic mode, it is the strength of the L2 regularization.
        - For stochastic mode, it is the strength of the KL divergence regularization.
    decoder
        Whether to use a decoder.
    pooling
        Pooling method, should be one of:

        - ``'mean'``: Aggregates combinations of covariates by the mean of their learned
          embeddings.
        - ``'attention_token'``: Aggregates combinations of covariates by an attention mechanism
          with a token.
        - ``'attention_seed'``: Aggregates combinations of covariates by an attention mechanism
          with a seed.
    pooling_kwargs
        Keyword arguments for the pooling method.
    covariates_not_pooled
        Covariates that will escape pooling (should be identical across all set elements).
    layers_before_pool
        Layers before pooling. Either a sequence of tuples with layer type and parameters or a
        dictionary with input-specific layers.
    layers_after_pool
        Layers after pooling.
    layers_decoder
        Layers for the decoder. Only relevant if ``'decoder'=True``.
    mask_value
        Value for masked elements used in input conditions.
    """

    output_dim: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 0.0
    decoder: bool = False
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=list)
    layers_before_pool: Layers_t | Layers_separate_input_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    layers_decoder: Layers_t = dc_field(default_factory=lambda: [])
    output_dropout: float = 0.0
    mask_value: float = 0.0

    def setup(self):
        """Initialize the modules."""
        # modules before pooling
        self.separate_inputs = isinstance(self.layers_before_pool, (dict | FrozenDict))
        if self.separate_inputs:
            # different layers for different inputs, before_pool_modules is of type Layers_separate_input_t
            self.before_pool_modules: dict[str, list[nn.Module]] | list[nn.Module] = {
                key: nn_utils._get_layers(layers)
                for key, layers in self.layers_before_pool.items()  # type: ignore[union-attr]
            }
        else:
            self.before_pool_modules = nn_utils._get_layers(self.layers_before_pool)  # type: ignore[arg-type]

        # pooling
        if self.pooling == "mean":
            self.pool_module = lambda x, mask, training: jnp.mean(x * mask, axis=-2)
        elif self.pooling == "attention_token":
            self.pool_module = nn_utils.TokenAttentionPooling(**self.pooling_kwargs)
        elif self.pooling == "attention_seed":
            self.pool_module = nn_utils.SeedAttentionPooling(**self.pooling_kwargs)

        # modules after pooling
        self.after_pool_modules_mean = nn_utils._get_layers(self.layers_after_pool, self.output_dim)

        if self.condition_mode == "stochastic":
            self.after_pool_modules_var = nn_utils._get_layers(self.layers_after_pool, self.output_dim)

    def __call__(
        self,
        conditions: dict[str, jnp.ndarray],
        training: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply the set encoder.

        Parameters
        ----------
        conditions : dict[str, jnp.ndarray]
            Dictionary of batch of conditions of shape ``(batch_size, set_size, condition_dim)``.
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        Mean and log-variance of conditions of shape ``(batch_size, output_dim)``.
        """
        mask, attention_mask = self._get_masks(conditions)

        # apply modules before pooling
        if self.separate_inputs:
            processed_inputs_pooling = []
            processed_inputs_other = []
            for pert_cov, conditions_i in conditions.items():
                # apply separate modules for all inputs
                conditions_i = nn_utils._apply_modules(
                    self.before_pool_modules[pert_cov],  # type: ignore[call-overload]
                    conditions_i,
                    attention_mask,
                    training,
                )
                if pert_cov in self.covariates_not_pooled:
                    # only keep first set element for covariates that are not pooled
                    processed_inputs_other.append(conditions_i[:, 0, :])
                else:
                    processed_inputs_pooling.append(conditions_i)

            conditions_pooling_arr = jnp.concatenate(processed_inputs_pooling, axis=-1)
            conditions_not_pooled = (
                jnp.concatenate(processed_inputs_other, axis=-1) if self.covariates_not_pooled else None
            )
        else:
            # by default, no modules before pooling for covariates that are not pooled
            if self.covariates_not_pooled:
                # divide conditions into pooled and not pooled
                conditions_not_pooled = []
                conditions_pooling = []
                for pert_cov in conditions:
                    if pert_cov in self.covariates_not_pooled:
                        conditions_not_pooled.append(conditions[pert_cov][:, 0, :])
                    else:
                        conditions_pooling.append(conditions[pert_cov])
                conditions_not_pooled = jnp.concatenate(
                    conditions_not_pooled,
                    axis=-1,
                )
                conditions_pooling_arr = jnp.concatenate(
                    conditions_pooling,
                    axis=-1,
                )

                # apply modules to pooled covariates
                conditions_pooling_arr = nn_utils._apply_modules(
                    self.before_pool_modules,  # type: ignore[arg-type]
                    conditions_pooling_arr,
                    attention_mask,
                    training,
                )
            else:
                conditions = jnp.concatenate(list(conditions.values()), axis=-1)
                conditions_pooling_arr = nn_utils._apply_modules(
                    self.before_pool_modules,
                    conditions,
                    attention_mask,
                    training,  # type: ignore[arg-type]
                )

        # pooling
        pool_mask = mask if self.pooling == "mean" else attention_mask
        conditions = self.pool_module(conditions_pooling_arr, pool_mask, training=training)
        if self.covariates_not_pooled:
            conditions = jnp.concatenate([conditions, conditions_not_pooled], axis=-1)

        # apply modules after pooling
        conditions = nn_utils._apply_modules(self.after_pool_modules_mean, conditions, None, training)

        if self.condition_mode == "stochastic":
            conditions_logvar = nn_utils._apply_modules(self.after_pool_modules_var, conditions, None, training)
        else:
            conditions_logvar = jnp.zeros_like(conditions)
        return conditions, conditions_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        conditions: dict[str, jnp.ndarray],
        **kwargs: Any,
    ):
        """Create initial training state."""
        params = self.init(
            rng,
            conditions={k: jnp.empty((1, v.shape[1], v.shape[2])) for k, v in conditions.items()},
            training=False,
        )["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            **kwargs,
        )

    def _get_masks(self, conditions: dict[str, ArrayLike]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get mask for padded conditions tensor."""
        # mask of shape (batch_size, set_size)
        mask = 1 - jnp.all(
            jnp.array(
                [jnp.all(c == self.mask_value, axis=-1) for c in conditions.values()],
            ),
            axis=0,
        )
        mask = jnp.expand_dims(mask, -1)

        # attention mask of shape (batch_size, 1, set_size, set_size)
        attention_mask = mask & jnp.matrix_transpose(mask)
        attention_mask = jnp.expand_dims(attention_mask, 1)

        return mask, attention_mask
