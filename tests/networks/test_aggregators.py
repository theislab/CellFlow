import jax
import jax.numpy as jnp
import pytest

from cellflow.networks._set_encoders import ConditionEncoder
from cellflow.networks._utils import SeedAttentionPooling, TokenAttentionPooling


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
