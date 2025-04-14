import abc
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from cellflow._types import Layers_t

__all__ = [
    "SelfAttention",
    "SeedAttentionPooling",
    "TokenAttentionPooling",
    "MLPBlock",
    "BaseModule",
]


class BaseModule(abc.ABC, nn.Module):
    """Base module for condition encoder and its components."""

    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass."""
        pass


class MLPBlock(BaseModule):
    """
    MLP block.

    Parameters
    ----------
    dims
        Dimensions of the MLP layers.
    dropout_rate
        Dropout rate.
    act_last_layer
        Whether to apply the activation function to the last layer.
    act_fn
        Activation function.
    """

    dims: Sequence[int] = (1024, 1024, 1024)
    dropout_rate: float = 0.0
    act_last_layer: bool = True
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch_size, input_dim)``.
        training
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape ``(batch_size, output_dim)``.
        """
        if len(self.dims) == 0:
            return x
        z = x
        for i in range(len(self.dims) - 1):
            z = self.act_fn(nn.Dense(self.dims[i])(z))
            z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
        z = nn.Dense(self.dims[-1])(z)
        z = self.act_fn(z) if self.act_last_layer else z
        z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
        return z


class FilmBlock(BaseModule):
    """Feature-wise Linear Modulation (FiLM) layer.

    Applies a learned affine transformation (scale and shift) to the input,
    conditioned on an external embedding.

    Parameters
    ----------
    input_dim
        Dimensionality of the input features.
    cond_dim
        Dimensionality of the conditioning features.
    act_fn
        Activation function to apply after modulation. If :obj:`None`, no activation is applied.
    """

    input_dim: int
    cond_dim: int
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x

    def setup(self) -> None:
        self.film_generator = nn.Dense(self.input_dim * 2)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """Applies FiLM modulation.

        Parameters
        ----------
        x
            Input features of shape (batch, input_dim).
        cond
            Conditioning features of shape (batch, cond_dim).

        Returns
        -------
            Modulated features of shape (batch, input_dim).
        """
        gamma_beta = self.film_generator(cond)  # shape: (batch, input_dim * 2)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)  # each shape: (batch, input_dim)
        return self.act_fn(gamma * x + beta)


class ResNetBlock(nn.Module):
    """Residual conditioning block.

    Applies a residual MLP transformation to the input, conditioned on external features.

    Parameters
    ----------
    input_dim
        Dimensionality of the input features.
    projection_dims
        Dimensionality of the projection layers.
    hidden_dims
        Hidden layer sizes for the residual block.
    act_fn
        Activation function to apply in the MLP block.
    dropout_rate
        Dropout rate applied after each hidden layer.
    """

    input_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    projection_dims: Sequence[int] = (256, 256)
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    dropout_rate: float = 0.0

    def setup(self):
        self.mlp_block_1 = MLPBlock(dims=self.hidden_dims, act_fn=self.act_fn, dropout_rate=self.dropout_rate)
        self.mlp_block_2 = MLPBlock(dims=self.hidden_dims, act_fn=self.act_fn, dropout_rate=self.dropout_rate)
        self.cond_proj = MLPBlock(dims=self.projection_dims, act_fn=self.act_fn, dropout_rate=self.dropout_rate)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, *, training: bool = True) -> jnp.ndarray:
        """Forward pass of the residual layer.

        Parameters
        ----------
        x
            Input features of shape (batch, input_dim).
        cond
            Conditioning features of shape (batch, cond_dim).
        training
            Whether the model is in training mode.

        Returns
        -------
            Output features of shape (batch, input_dim).
        """
        h = self.mlp_block_1(x)
        h = h + self.cond_proj(cond)
        h = self.mlp_block_2(h)
        return h


class SelfAttention(BaseModule):
    """Self-attention layer

    Self-attention layer that can optionally be followed by a FC layer with residual connection,
    making it a transformer block.

    Parameters
    ----------
    num_heads
        Number of heads.
    qkv_dim
        Dimensionality of the query, key, and value.
    dropout_rate
        Dropout rate.
    transformer_block
        Whether to make it a transformer block (adds FC layer with residual connection).
    layer_norm
        Whether to use layer normalization
    """

    num_heads: int = 8
    qkv_dim: int = 64
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ):
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch_size, set_size, input_dim)`` or
            ``(batch_size, input_dim)``.
        mask
            Mask tensor of shape ``(batch_size, 1 | num_heads, set_size, set_size)``.
        training
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape ``(batch_size, set_size, input_dim)``.
        """
        squeeze = x.ndim == 2
        x = jnp.expand_dims(x, 1) if squeeze else x

        # self-attention
        z = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_dim,
            dropout_rate=self.dropout_rate,
        )(x, mask=mask, deterministic=not training)

        if self.transformer_block:
            # query residual connection
            z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
            z = z + x
            if self.layer_norm:
                z = nn.LayerNorm()(z)
            # FC layer with residual connection
            z_ = self.act_fn(nn.Dense(self.qkv_dim)(z))
            z_ = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
            z = z + z_

        return z.squeeze(1) if squeeze else z


class SelfAttentionBlock(BaseModule):
    """
    Several self-attention (+ optional FC layer) layers stacked together.

    Parameters
    ----------
    num_heads
        Number of heads for each layer.
    qkv_dim
        Dimensionality of the query, key, and value for each layer.
    dropout_rate
        Dropout rate.
    transformer_block
        Whether to make layers transformer blocks (adds FC layer with residual connection).
    layer_norm
        Whether to use layer normalization.

    Returns
    -------
    Output tensor of shape (batch_size, set_size, input_dim).
    """

    num_heads: Sequence[int] | int
    qkv_dim: Sequence[int] | int
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    def __post_init__(self) -> None:
        """Initialize the module."""
        super().__post_init__()
        if not isinstance(self.num_heads, Sequence):
            self.num_heads = [self.num_heads]
        if not isinstance(self.qkv_dim, Sequence):
            self.qkv_dim = [self.qkv_dim]
        if len(self.num_heads) != len(self.qkv_dim):
            raise ValueError("The number of specified layers should be the same for num_heads and qkv_dims.")

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch_size, set_size, input_dim).
        mask : Optional[jnp.ndarray]
            Mask tensor of shape (batch_size, 1 | num_heads, set_size, set_size).
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape (batch_size, set_size, input_dim).
        """
        z = x
        for num_heads, qkv_dim in zip(self.num_heads, self.qkv_dim, strict=False):  # type: ignore[arg-type]
            z = SelfAttention(
                num_heads=num_heads,
                qkv_dim=qkv_dim,
                dropout_rate=self.dropout_rate,
                transformer_block=self.transformer_block,
                layer_norm=self.layer_norm,
                act_fn=self.act_fn,
            )(z, mask, training)
        return z


class SeedAttentionPooling(BaseModule):
    """
    Pooling by multi-head attention with a trainable seed.

    Parameters
    ----------
    num_heads
        Number of heads.
    v_dim
        Dimensionality of the value.
    seed_dim
        Dimensionality of the seed.
    dropout_rate
        Dropout rate.
    transformer_block
        Whether to make it a transformer block (adds FC layer with residual connection).
    layer_norm
        Whether to use layer normalization.
    act_fn
        Activation function.

    References
    ----------
    :cite:`vaswani:17`
    """

    num_heads: int = 8
    v_dim: int = 64
    seed_dim: int = 64
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ):
        """
        Apply the pooling by multi-head attention.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch_size, set_size, input_dim)``.
        mask
            Mask tensor of shape ``(batch_size, 1, set_size, set_size)``.
        training
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape ``(batch_size, input_dim)``.
        """
        # trainable seed
        S = self.param("S", initializers.xavier_uniform(), (1, 1, self.seed_dim))
        S = jnp.tile(S, (x.shape[0], 1, 1))

        # multi-head attention
        Q = nn.Dense(self.v_dim)(S)
        K, V = nn.Dense(self.v_dim)(x), nn.Dense(self.v_dim)(x)
        Q_ = jnp.concatenate(jnp.split(Q, self.num_heads, axis=2), axis=0)
        K_ = jnp.concatenate(jnp.split(K, self.num_heads, axis=2), axis=0)
        V_ = jnp.concatenate(jnp.split(V, self.num_heads, axis=2), axis=0)
        A = jnp.matmul(Q_, K_.transpose(0, 2, 1)) / jnp.sqrt(self.v_dim)
        A = jnp.matmul(Q_, K_.transpose(0, 2, 1)) / jnp.sqrt(self.v_dim)
        if mask is not None:
            # mask from (batch_, 1 | num_heads, set_, set_) to (batch_ * num_heads, 1, set_)
            mask = jnp.repeat(mask[:, 0, [0], :], self.num_heads, axis=0)
            A = jnp.where(mask, A, -1e9)
        A = nn.softmax(A)
        A = jnp.matmul(A, V_)

        if self.transformer_block:
            # query residual connection
            O = jnp.concatenate(jnp.split(Q_ + A, self.num_heads, axis=0), axis=2)
            O = nn.Dropout(rate=self.dropout_rate)(O, deterministic=not training)
            if self.layer_norm:
                O = nn.LayerNorm()(O)
            # FC layer with residual connection
            O_ = self.act_fn(nn.Dense(self.v_dim)(O))
            O_ = nn.Dropout(rate=self.dropout_rate)(O_, deterministic=not training)
            O = O + O_
            if self.layer_norm:
                O = nn.LayerNorm()(O)
        else:
            O = jnp.concatenate(jnp.split(A, self.num_heads, axis=0), axis=2)

        return O.squeeze(1)


class TokenAttentionPooling(BaseModule):
    """
    Multi-head attention which aggregates sets by learning a token.

    Parameters
    ----------
    num_heads
        Number of attention heads.
    qkv_dim
        Dimensionality of the query, key, and value.
    dropout_rate
        Dropout rate.
    act_fn
        Activation function.
    """

    num_heads: int = 8
    qkv_dim: int = 64
    dropout_rate: float = 0.0
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape (batch_size, set_size, input_dim).
        mask
            Mask tensor of shape (batch_size, 1 | num_heads, set_size, set_size).
        training
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape ``(batch_size, input_dim)``.
        """
        # add token
        token_shape = (len(x), 1)
        class_token = nn.Embed(num_embeddings=1, features=x.shape[-1])(jnp.int32(jnp.zeros(token_shape)))
        z = jnp.concatenate((class_token, x), axis=-2)
        token_mask = jnp.zeros((x.shape[0], 1, x.shape[1] + 1, x.shape[1] + 1))
        token_mask = token_mask.at[:, :, 0, :].set(1)
        token_mask = token_mask.at[:, :, :, 0].set(1)
        token_mask = token_mask.at[:, :, 1:, 1:].set(mask)

        # attention
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_dim,
            dropout_rate=self.dropout_rate,
        )
        emb = attention(z, mask=token_mask, deterministic=not training)

        # only continue with token 0
        z = emb[:, 0, :]

        return z


def _get_layers(
    layers: Layers_t,
    output_dim: int | None = None,
    dropout_rate: float | None = None,
) -> list[nn.Module]:
    """Get modules from layer parameters."""
    modules = []
    if isinstance(layers, Sequence):
        for layer in layers:
            layer = dict(layer)
            layer_type = layer.pop("layer_type", "mlp")
            if layer_type == "mlp":
                lay = MLPBlock(**layer)
            elif layer_type == "self_attention":
                lay = SelfAttentionBlock(**layer)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            modules.append(lay)
    if output_dim is not None:
        modules.append(nn.Dense(output_dim))
        if dropout_rate is not None:
            modules.append(nn.Dropout(dropout_rate))
    return modules


def _apply_modules(
    modules: list[nn.Module],
    conditions: jax.Array,
    attention_mask: jnp.ndarray | None,
    training: bool,
) -> jnp.ndarray:
    """Apply modules to conditions."""
    for module in modules:
        if isinstance(module, SelfAttentionBlock):
            conditions = module(conditions, attention_mask, training)
        elif isinstance(module, nn.Dense):
            conditions = module(conditions)
        elif isinstance(module, nn.Dropout):
            conditions = module(conditions, deterministic=not training)
        else:
            conditions = module(conditions, training)
    return conditions
