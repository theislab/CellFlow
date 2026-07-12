import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from cellflow._types import Layers_separate_input_t, Layers_t
from cellflow.networks._set_encoders import ConditionEncoder
from cellflow.networks._utils import FilmBlock, MLPBlock, ResNetBlock, sinusoidal_time_encoder

__all__ = [
    "ConditionalVelocityField",
    "GENOTConditionalVelocityField",
    "null_condition_embedding",
    "null_condition_input",
]


def null_condition_embedding(
    cond_embedding: jnp.ndarray,
    *,
    condition_dropout_prob: float,
    make_rng: Callable[[str], jax.Array],
    train: bool,
    force_uncond: bool,
) -> jnp.ndarray:
    """Null the condition *embedding* for classifier-free guidance (``condition_null='zero_embedding'``).

    - If ``force_uncond`` is set, the condition is always dropped, yielding the
      unconditional velocity field (used at inference time).
    - Otherwise, during training the condition is dropped independently per set
      element with probability ``condition_dropout_prob``.

    With ``force_uncond=False`` and ``condition_dropout_prob == 0.0`` this is a no-op
    that reproduces the standard conditional behavior byte-for-byte: no random number
    is drawn and the embedding is returned as-is. ``make_rng`` is the owning module's
    :meth:`flax.linen.Module.make_rng` (drawn only when a dropout mask is needed).
    """
    if force_uncond:
        return jnp.zeros_like(cond_embedding)
    if train and condition_dropout_prob > 0.0:
        keep = jax.random.bernoulli(
            make_rng("dropout"),
            p=1.0 - condition_dropout_prob,
            shape=(cond_embedding.shape[0], 1),
        )
        return jnp.where(keep, cond_embedding, jnp.zeros_like(cond_embedding))
    return cond_embedding


def null_condition_input(
    cond: dict[str, jnp.ndarray],
    *,
    condition_dropout_prob: float,
    mask_value: float,
    make_rng: Callable[[str], jax.Array],
    train: bool,
    force_uncond: bool,
) -> dict[str, jnp.ndarray]:
    """Null the *raw* condition by filling it with ``mask_value`` (``condition_null='mask_value'``).

    Routes an all-masked condition set through the condition encoder, so the
    unconditional representation is whatever the encoder maps a fully-masked set to
    (matching how padded conditions are handled). Same drop policy as
    :func:`null_condition_embedding`: always when ``force_uncond`` is set, otherwise per
    set element with probability ``condition_dropout_prob`` during training. With the
    defaults it returns ``cond`` unchanged and draws no random number.
    """
    if force_uncond:
        return jax.tree_util.tree_map(lambda c: jnp.full_like(c, mask_value), cond)
    if train and condition_dropout_prob > 0.0:
        n = next(iter(cond.values())).shape[0]
        keep = jax.random.bernoulli(make_rng("dropout"), p=1.0 - condition_dropout_prob, shape=(n, 1, 1))
        return jax.tree_util.tree_map(lambda c: jnp.where(keep, c, jnp.full_like(c, mask_value)), cond)
    return cond


class ConditionalVelocityField(nn.Module):
    """Parameterized neural vector field with conditions.

    Parameters
    ----------
        output_dim
            Dimensionality of the output.
        max_combination_length
            Maximum number of covariates in a combination.
        condition_mode
            Mode of the encoder, should be one of:

            - ``'deterministic'``: Learns condition encoding point-wise.
            - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.

        regularization
            Regularization strength in the latent space:

            - For deterministic mode, it is the strength of the L2 regularization.
            - For stochastic mode, it is the strength of the KL divergence regularization.

        condition_embedding_dim
            Dimensions of the condition embedding.
        covariates_not_pooled
            Covariates that will escape pooling (should be identical across all set elements).
        pooling
            Pooling method.
        pooling_kwargs
            Keyword arguments for the pooling method.
        layers_before_pool
            Layers before pooling. Either a sequence of tuples with layer type and parameters or
            a dictionary with input-specific layers.
        layers_after_pool
            Layers after pooling.
        cond_output_dropout
            Dropout rate for the last layer of the condition encoder.
        condition_dropout_prob
            Probability of dropping the whole condition during training so that the
            network also learns an unconditional velocity field. This enables
            classifier-free guidance at inference time via the ``force_uncond``
            argument of :meth:`__call__`. A value of ``0.0`` (the default) disables
            condition dropout and reproduces the standard conditional behavior.
        condition_null
            How the condition is nulled when it is dropped (both during training via
            ``condition_dropout_prob`` and at inference via ``force_uncond``):

            - ``'zero_embedding'``: zero the condition embedding after the encoder
              (the default).
            - ``'mask_value'``: fill the raw condition inputs with :attr:`mask_value`
              and route the resulting fully-masked set through the encoder.

        condition_encoder_kwargs
            Keyword arguments for the condition encoder.
        act_fn
            Activation function.
        time_freqs
            Frequency of the cyclical time encoding.
        time_max_period
            Controls the minimum frequency of the time embeddings.
        time_encoder_dims
            Dimensions of the time embedding.
        time_encoder_dropout
            Dropout rate for the time embedding.
        hidden_dims
            Dimensions of the hidden layers.
        hidden_dropout
            Dropout rate for the hidden layers.
        conditioning
            Conditioning method, should be one of:

            - ``'concatenation'``: Concatenate the time, data, and condition embeddings.
            - ``'film'``: Use FiLM conditioning, i.e. learn FiLM weights from time and condition embedding
              to scale the data embeddings.
            - ``'resnet'``: Use residual conditioning.

        conditioning_kwargs
            Keyword arguments for the conditioning method.
        decoder_dims
            Dimensions of the output layers.
        decoder_dropout
            Dropout rate for the output layers.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data.

    Returns
    -------
        Output of the neural vector field.
    """

    output_dim: int
    max_combination_length: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 1.0
    condition_embedding_dim: int = 32
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=lambda: [])
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    cond_output_dropout: float = 0.0
    condition_dropout_prob: float = 0.0
    condition_null: Literal["zero_embedding", "mask_value"] = "zero_embedding"
    mask_value: float = 0.0
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_max_period: int = 10000
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    conditioning: Literal["concatenation", "film", "resnet"] = "concatenation"
    conditioning_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    @staticmethod
    def _normalize_vf_kwargs(vf_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        """Validate/normalize the solver-specific ``vf_kwargs`` this velocity field is built with.

        Called by :meth:`cellflow.model.CellFlow.prepare_model` so each velocity field owns which extra
        constructor kwargs it accepts, keeping the model code free of per-solver branches. The plain
        conditional field takes none.
        """
        if vf_kwargs is not None:
            raise ValueError("For `solver='otfm'`, `vf_kwargs` must be `None`.")
        return {}

    def setup(self):
        """Initialize the network."""
        if isinstance(self.conditioning_kwargs, dataclasses.Field):
            conditioning_kwargs = dict(self.conditioning_kwargs.default_factory())
        else:
            conditioning_kwargs = dict(self.conditioning_kwargs)
        self.condition_encoder = ConditionEncoder(
            condition_mode=self.condition_mode,
            regularization=self.regularization,
            output_dim=self.condition_embedding_dim,
            pooling=self.pooling,
            pooling_kwargs=self.pooling_kwargs,
            layers_before_pool=self.layers_before_pool,
            layers_after_pool=self.layers_after_pool,
            covariates_not_pooled=self.covariates_not_pooled,
            mask_value=self.mask_value,
            **self.condition_encoder_kwargs,
        )

        self.layer_cond_output_dropout = nn.Dropout(rate=self.cond_output_dropout)
        self.layer_norm_condition = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

        self._setup_conditioning(conditioning_kwargs)

    def _setup_conditioning(self, conditioning_kwargs: dict[str, Any]) -> None:
        """Build the ``conditioning``-mode-specific submodules.

        Called at the end of :meth:`setup` (so all shared submodules already exist on
        ``self``). Override in a subclass to add new ``conditioning`` modes, delegating to
        ``super()._setup_conditioning`` for the built-in ones.
        """
        if self.conditioning == "film":
            self.film_block = FilmBlock(
                input_dim=self.hidden_dims[-1],
                cond_dim=self.time_encoder_dims[-1] + self.condition_embedding_dim,
                **conditioning_kwargs,
            )
        elif self.conditioning == "resnet":
            self.resnet_block = ResNetBlock(
                input_dim=self.hidden_dims[-1],
                **conditioning_kwargs,
            )
        elif self.conditioning == "concatenation":
            if len(conditioning_kwargs) > 0:
                raise ValueError("If `conditioning=='concatenation' mode, no conditioning kwargs can be passed.")
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}")

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
        force_uncond: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        squeeze = x_t.ndim == 1
        if self.condition_null == "mask_value":
            cond = self._maybe_null_input(cond, train=train, force_uncond=force_uncond)
        cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
        if self.condition_mode == "deterministic":
            cond_embedding = cond_mean
        else:
            cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)

        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)
        if self.condition_null == "zero_embedding":
            cond_embedding = self._maybe_null_embedding(cond_embedding, train=train, force_uncond=force_uncond)

        t_encoded = sinusoidal_time_encoder(t, time_freqs=self.time_freqs, time_max_period=self.time_max_period)
        t_encoded = self.time_encoder(t_encoded, training=train)
        x_encoded = self._encode_x(x_t, squeeze, train)

        t_encoded = self.layer_norm_time(t_encoded)
        x_encoded = self.layer_norm_x(x_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)  # , 0)
        elif cond_embedding.shape[0] != x_t.shape[0]:  # type: ignore[attr-defined]
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        out = self._combine_and_decode(t_encoded, x_encoded, cond_embedding, squeeze, train)
        return out, cond_mean, cond_logvar

    def _encode_x(self, x_t: jnp.ndarray, squeeze: bool, train: bool) -> jnp.ndarray:
        """Encode ``x_t`` before conditioning. Override to insert pre-conditioning processing."""
        return self.x_encoder(x_t, training=train)

    def _conditioning_signals(
        self,
        t_encoded: jnp.ndarray,
        x_encoded: jnp.ndarray,
        cond_embedding: jnp.ndarray,
        x_0_encoded: jnp.ndarray | None = None,
    ) -> tuple[tuple[jnp.ndarray, ...], jnp.ndarray]:
        """Return ``(concat_inputs, conditioning_vec)`` for the conditioning step.

        ``concat_inputs`` is what ``'concatenation'`` concatenates (order matters for
        checkpoint compatibility); ``conditioning_vec`` is what modulates ``x`` for
        ``'film'``/``'resnet'`` (and ``'adaln_zero'`` in subclasses). ``GENOT`` folds the
        encoded source ``x_0`` into both.
        """
        if x_0_encoded is None:
            concat_inputs = (t_encoded, x_encoded, cond_embedding)
            conditioning_vec = jnp.concatenate((t_encoded, cond_embedding), axis=-1)
        else:
            concat_inputs = (t_encoded, x_encoded, x_0_encoded, cond_embedding)
            conditioning_vec = jnp.concatenate((t_encoded, x_0_encoded, cond_embedding), axis=-1)
        return concat_inputs, conditioning_vec

    def _combine_and_decode(
        self,
        t_encoded: jnp.ndarray,
        x_encoded: jnp.ndarray,
        cond_embedding: jnp.ndarray,
        squeeze: bool,
        train: bool,
        x_0_encoded: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Combine ``(t, x, [x_0,] condition)`` per the conditioning mode, decode, and project.

        Override in a subclass to add new ``conditioning`` modes, delegating to
        ``super()._combine_and_decode`` for the built-in ones. ``x_0_encoded`` is the encoded
        GENOT source (folded into the conditioning by :meth:`_conditioning_signals`).
        """
        concat_inputs, conditioning_vec = self._conditioning_signals(t_encoded, x_encoded, cond_embedding, x_0_encoded)
        if self.conditioning == "concatenation":
            out = jnp.concatenate(concat_inputs, axis=-1)
        elif self.conditioning == "film":
            out = self.film_block(x_encoded, conditioning_vec)
        elif self.conditioning == "resnet":
            out = self.resnet_block(x_encoded, conditioning_vec)
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}.")

        out = self.decoder(out, training=train)
        return self.output_layer(out)

    def _maybe_null_embedding(
        self,
        cond_embedding: jnp.ndarray,
        train: bool,
        force_uncond: bool,
    ) -> jnp.ndarray:
        """Null the condition embedding for classifier-free guidance (``condition_null='zero_embedding'``).

        - If ``force_uncond`` is set, the condition is always dropped, yielding the
          unconditional velocity field (used at inference time).
        - Otherwise, during training the whole condition is dropped independently per
          set element with probability :attr:`condition_dropout_prob`.

        With ``force_uncond=False`` and ``condition_dropout_prob == 0.0`` (the
        defaults) this is a no-op that reproduces the standard conditional behavior
        byte-for-byte: no random number is drawn and the embedding is returned as-is.
        """
        return null_condition_embedding(
            cond_embedding,
            condition_dropout_prob=self.condition_dropout_prob,
            make_rng=self.make_rng,
            train=train,
            force_uncond=force_uncond,
        )

    def _maybe_null_input(
        self,
        cond: dict[str, jnp.ndarray],
        train: bool,
        force_uncond: bool,
    ) -> dict[str, jnp.ndarray]:
        """Null the raw condition by filling it with :attr:`mask_value` (``condition_null='mask_value'``).

        This routes an all-masked condition set through the condition encoder, so the
        unconditional representation is whatever the encoder maps a fully-masked set
        to (matching :mod:`cellflow`'s handling of padded conditions). It mirrors the
        drop policy of :meth:`_maybe_null_embedding`: always dropped when
        ``force_uncond`` is set, otherwise dropped per set element with probability
        :attr:`condition_dropout_prob` during training. With the defaults it returns
        ``cond`` unchanged and draws no random number.
        """
        return null_condition_input(
            cond,
            condition_dropout_prob=self.condition_dropout_prob,
            mask_value=self.mask_value,
            make_rng=self.make_rng,
            train=train,
            force_uncond=force_uncond,
        )

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the embedding of the condition.

        Parameters
        ----------
            condition
                Conditioning vector of shape ``[batch, ...]``.

        Returns
        -------
            Learnt mean and log-variance of the condition embedding.
            If :attr:`cellflow.model.CellFlow.condition_mode` is ``'deterministic'``, the log-variance
            is set to zero.
        """
        condition_mean, condition_logvar = self.condition_encoder(condition, training=False)
        return condition_mean, condition_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create the training state.

        Parameters
        ----------
            rng
                Random number generator.
            optimizer
                Optimizer.
            input_dim
                Dimensionality of the velocity field.
            conditions
                Conditions describing the perturbation.

        Returns
        -------
            The training state.
        """
        t, x_t = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        params_rng, condition_encoder_rng = jax.random.split(rng, 2)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng},
            t=t,
            x_t=x_t,
            cond=cond,
            encoder_noise=encoder_noise,
            train=False,
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

    @property
    def output_dims(self):
        """Dimensions of the output layers."""
        return tuple(self.decoder_dims) + (self.output_dim,)

    @property
    def time_encoder(self):
        """The time encoder used."""
        return self._time_encoder

    @time_encoder.setter
    def time_encoder(self, encoder):
        """Set the time encoder."""
        self._time_encoder = encoder

    @property
    def x_encoder(self):
        """The x encoder used."""
        return self._x_encoder

    @x_encoder.setter
    def x_encoder(self, encoder):
        """Set the x encoder."""
        self._x_encoder = encoder

    @property
    def decoder(self):
        """The decoder used."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        """Set the decoder."""
        self._decoder = decoder


class GENOTConditionalVelocityField(ConditionalVelocityField):
    """Parameterized neural vector field with conditions for GENOT.

    Parameters
    ----------
        output_dim
            Dimensionality of the output.
        max_combination_length
            Maximum number of covariates in a combination.
        condition_mode
            Mode of the encoder, should be one of:

            - ``'deterministic'``: Learns condition encoding point-wise.
            - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.

        regularization
            Regularization strength in the latent space:

            - For deterministic mode, it is the strength of the L2 regularization.
            - For stochastic mode, it is the strength of the KL divergence regularization.

        condition_embedding_dim
            Dimensions of the condition embedding.
        covariates_not_pooled
            Covariates that will escape pooling (should be identical across all set elements).
        pooling
            Pooling method.
        pooling_kwargs
            Keyword arguments for the pooling method.
        layers_before_pool
            Layers before pooling. Either a sequence of tuples with layer type and parameters or
            a dictionary with input-specific layers.
        layers_after_pool
            Layers after pooling.
        cond_output_dropout
            Dropout rate for the last layer of the condition encoder.
        condition_encoder_kwargs
            Keyword arguments for the condition encoder.
        act_fn
            Activation function.
        time_freqs
            Frequency of the cyclical time encoding.
        time_max_period
            Controls the minimum frequency of the time embeddings.
        time_encoder_dims
            Dimensions of the time embedding.
        time_encoder_dropout
            Dropout rate for the time embedding.
        hidden_dims
            Dimensions of the hidden layers.
        hidden_dropout
            Dropout rate for the hidden layers.
        conditioning
            Conditioning method, should be one of:

            - ``'concatenation'``: Concatenate the time, data, and condition embeddings.
            - ``'film'``: Use FiLM conditioning, i.e. learn FiLM weights from time, x_0, and condition embedding
              to scale the data embeddings.
            - ``'resnet'``: Use residual conditioning.

        conditioning_kwargs
            Keyword arguments for the conditioning method.
        decoder_dims
            Dimensions of the output layers.
        decoder_dropout
            Dropout rate for the output layers.
        genot_source_dims
            Dimensions of the layers processing the source cells.
        genot_source_dropout
            Dropout rate for the layers processing the source cells.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data.

    Returns
    -------
        Output of the neural vector field.
    """

    output_dim: int
    max_combination_length: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 1.0
    condition_embedding_dim: int = 32
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=lambda: [])
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    cond_output_dropout: float = 0.0
    mask_value: float = 0.0
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_max_period: int = 10000
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    conditioning: Literal["concatenation", "film", "resnet"] = "concatenation"
    conditioning_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0
    genot_source_dims: Sequence[int] = (1024, 1024, 1024)
    genot_source_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    @staticmethod
    def _normalize_vf_kwargs(vf_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        """GENOT's velocity field needs source-processing kwargs; default them when not given.

        See :meth:`ConditionalVelocityField._normalize_vf_kwargs`. ``GENOT`` requires
        ``genot_source_dims`` and ``genot_source_dropout``.
        """
        if vf_kwargs is None:
            return {"genot_source_dims": [1024, 1024, 1024], "genot_source_dropout": 0.0}
        if not isinstance(vf_kwargs, dict):
            raise TypeError(f"`vf_kwargs` must be a dict or None, got {type(vf_kwargs).__name__}.")
        allowed = {"genot_source_dims", "genot_source_dropout"}
        unknown = set(vf_kwargs) - allowed
        if unknown:
            raise ValueError(f"Unexpected `vf_kwargs` keys {sorted(unknown)}; allowed: {sorted(allowed)}.")
        missing = allowed - set(vf_kwargs)
        if missing:
            raise ValueError(f"Missing `vf_kwargs` keys {sorted(missing)}; required: {sorted(allowed)}.")
        return vf_kwargs

    def setup(self):
        """Initialize the network."""
        if isinstance(self.conditioning_kwargs, dataclasses.Field):
            conditioning_kwargs = dict(self.conditioning_kwargs.default_factory())
        else:
            conditioning_kwargs = dict(self.conditioning_kwargs)
        self.condition_encoder = ConditionEncoder(
            condition_mode=self.condition_mode,
            regularization=self.regularization,
            output_dim=self.condition_embedding_dim,
            pooling=self.pooling,
            pooling_kwargs=self.pooling_kwargs,
            layers_before_pool=self.layers_before_pool,
            layers_after_pool=self.layers_after_pool,
            output_dropout=self.cond_output_dropout,
            covariates_not_pooled=self.covariates_not_pooled,
            mask_value=self.mask_value,
            **self.condition_encoder_kwargs,
        )
        self.layer_cond_output_dropout = nn.Dropout(rate=self.cond_output_dropout)
        self.layer_norm_condition = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_0_encoder = MLPBlock(
            dims=self.genot_source_dims,
            act_fn=self.act_fn,
            dropout_rate=self.genot_source_dropout,
        )
        self.layer_norm_x_0 = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

        self._setup_conditioning(conditioning_kwargs)

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        x_0: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ):
        squeeze = x_t.ndim == 1
        cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
        if self.condition_mode == "deterministic":
            cond_embedding = cond_mean
        else:
            cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)
        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)
        t_encoded = sinusoidal_time_encoder(t, time_freqs=self.time_freqs, time_max_period=self.time_max_period)
        t_encoded = self.time_encoder(t_encoded, training=train)
        x_encoded = self.x_encoder(x_t, training=train)
        x_0_encoded = self.x_0_encoder(x_0, training=train)

        t_encoded = self.layer_norm_time(t_encoded)
        x_encoded = self.layer_norm_x(x_encoded)
        x_0_encoded = self.layer_norm_x_0(x_0_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)  # , 0)
        elif cond_embedding.shape[0] != x_t.shape[0]:  # type: ignore[attr-defined]
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        out = self._combine_and_decode(t_encoded, x_encoded, cond_embedding, squeeze, train, x_0_encoded=x_0_encoded)
        return out, cond_mean, cond_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create the training state.

        Parameters
        ----------
            rng
                Random number generator.
            optimizer
                Optimizer.
            input_dim
                Dimensionality of the velocity field.
            conditions
                Conditions describing the perturbation.

        Returns
        -------
            The training state.
        """
        t, x_t, x_0 = jnp.ones((1, 1)), jnp.ones((1, input_dim)), jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        params_rng, condition_encoder_rng = jax.random.split(rng, 2)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng},
            t=t,
            x_t=x_t,
            x_0=x_0,
            cond=cond,
            encoder_noise=encoder_noise,
            train=False,
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
