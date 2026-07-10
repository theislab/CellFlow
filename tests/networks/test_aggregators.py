import jax
import jax.numpy as jnp
import pytest

from cellflow.networks._set_encoders import ConditionEncoder
from cellflow.networks._utils import (
    SeedAttentionPooling,
    SelfAttention,
    SelfAttentionBlock,
    TokenAttentionPooling,
)


class TestAggregator:
    @pytest.mark.parametrize("agg", [TokenAttentionPooling, SeedAttentionPooling])
    def test_mask_impact_on_TokenAttentionPooling(self, agg):
        rng = jax.random.PRNGKey(0)
        init_rng, mask_rng = jax.random.split(rng, 2)
        condition = jax.random.normal(rng, (2, 3, 7))
        condition = jnp.concatenate((condition, jnp.zeros((2, 1, 7))), axis=1)
        cond_encoder = ConditionEncoder(32)
        _, attn_mask = cond_encoder._get_masks({"conditions": condition})
        random_mask = jax.random.bernoulli(mask_rng, 0.5, attn_mask.shape).astype(jnp.int32)
        agg = agg()
        variables = agg.init(init_rng, condition, random_mask, training=True)
        out = agg.apply(variables, condition, attn_mask, training=True)
        out_rand = agg.apply(variables, condition, random_mask, training=True)
        # output dim = input dim for TokenAttentionPooling, output dim = 64 by default in SeedAttentionPooling
        assert out.shape[0] == 2
        assert out.shape[1] == 7 if isinstance(agg, TokenAttentionPooling) else 64
        assert out_rand.shape[0] == 2
        assert out_rand.shape[1] == 7 if isinstance(agg, TokenAttentionPooling) else 64
        assert not jnp.allclose(out[0], out_rand[0], atol=1e-6)
        assert not jnp.allclose(out[1], out_rand[1], atol=1e-6)


class TestTransformerEncoder:
    """Tests for the transformer-style enhancements to the attention encoder."""

    def test_self_attention_default_is_pure_attention(self):
        """``transformer_block=False`` (the default) must stay the original single self-attention
        layer: no feed-forward ``Dense``/``LayerNorm`` params are created, so the behavior and the
        Flax param tree are unchanged vs before this feature was added."""
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 5, 16))
        mask = jnp.ones((2, 1, 5, 5))
        sa = SelfAttention()  # defaults: transformer_block=False
        params = sa.init(rng, x, mask, training=False)["params"]
        assert set(params.keys()) == {"MultiHeadDotProductAttention_0"}
        out = sa.apply({"params": params}, x, mask, training=False)
        assert out.shape == (2, 5, 16)

    def test_token_pooling_default_param_structure(self):
        """Default ``TokenAttentionPooling`` (``num_layers=1``, ``transformer_block=False``) keeps the
        historical param tree (``Embed_0`` + ``MultiHeadDotProductAttention_0``) so pre-refactor
        checkpoints restore unchanged."""
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 4, 16))
        mask = jnp.ones((2, 1, 4, 4))
        tap = TokenAttentionPooling()  # defaults: num_layers=1, transformer_block=False
        params = tap.init(rng, x, mask, training=False)["params"]
        assert set(params.keys()) == {"Embed_0", "MultiHeadDotProductAttention_0"}
        out = tap.apply({"params": params}, x, mask, training=False)
        assert out.shape == (2, 16)

    def test_self_attention_ff_dim_expansion(self):
        """The position-wise FFN expands to ``ff_dim`` and projects back to the input dim; the default
        expansion is 4x the input dim (NOT ``qkv_dim``)."""
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 5, 16))
        mask = jnp.ones((2, 1, 5, 5))

        # explicit ff_dim -> FFN is [Dense(ff_dim), Dense(input_dim)] regardless of qkv_dim
        sa = SelfAttention(qkv_dim=8, ff_dim=40, transformer_block=True)
        params = sa.init(rng, x, mask, training=False)["params"]
        dense_out = sorted(v["kernel"].shape[-1] for k, v in params.items() if k.startswith("Dense"))
        assert dense_out == [16, 40]

        # default ff_dim -> 4 * input_dim = 64 (not qkv_dim=8)
        sa_def = SelfAttention(qkv_dim=8, transformer_block=True)
        params_def = sa_def.init(rng, x, mask, training=False)["params"]
        dense_out_def = sorted(v["kernel"].shape[-1] for k, v in params_def.items() if k.startswith("Dense"))
        assert dense_out_def == [16, 64]

    @pytest.mark.parametrize("ff_dim", [None, 128])
    @pytest.mark.parametrize("layer_norm", [False, True])
    def test_self_attention_block_transformer_forward(self, ff_dim, layer_norm):
        """A stacked ``SelfAttentionBlock`` transformer with per-layer heads/dims and an ``ff_dim``
        builds and runs a forward pass, preserving the input dim."""
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 5, 16))
        mask = jnp.ones((2, 1, 5, 5))
        block = SelfAttentionBlock(
            num_heads=[4, 8],
            qkv_dim=[32, 64],
            ff_dim=ff_dim,
            transformer_block=True,
            layer_norm=layer_norm,
        )
        params = block.init(rng, x, mask, training=False)
        out = block.apply(params, x, mask, training=False)
        assert out.shape == (2, 5, 16)

    @pytest.mark.parametrize("transformer_block", [False, True])
    def test_token_pooling_multilayer_forward(self, transformer_block):
        """Multi-layer ``TokenAttentionPooling`` takes the opt-in ``SelfAttentionBlock`` path (no
        top-level ``MultiHeadDotProductAttention_0``) and reads out the CLS token."""
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 4, 16))
        mask = jnp.ones((2, 1, 4, 4))
        tap = TokenAttentionPooling(
            num_layers=2,
            transformer_block=transformer_block,
            layer_norm=True,
            ff_dim=32,
        )
        params = tap.init(rng, x, mask, training=False)["params"]
        assert any("SelfAttentionBlock" in k for k in params)
        assert "MultiHeadDotProductAttention_0" not in params
        out = tap.apply({"params": params}, x, mask, training=False)
        assert out.shape == (2, 16)
