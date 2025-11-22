import jax
import jax.numpy as jnp
import optax
import pytest

import scaleflow

cond = {
    "pert1": jnp.ones((1, 3, 3)),
    "pert2": jnp.ones((1, 3, 10)),
    "pert3": jnp.ones((1, 3, 5)),
}
cond = {k: v.at[0, 2, :].set(0.0) for k, v in cond.items()}
cond["pert4_skip_pool"] = jnp.ones((1, 3, 5))

layers_before_pool = [
    {
        "pert1": (
            {"layer_type": "mlp", "dims": (32, 32)},
            {"layer_type": "self_attention", "num_heads": 4, "qkv_dim": 32},
        ),
        "pert2": ({"layer_type": "mlp", "dims": (32, 32)},),
        "pert3": (),
        "pert4_skip_pool": (),
    },
    ({"layer_type": "mlp", "dims": (32, 32)},),
    (
        {
            "layer_type": "self_attention",
            "num_heads": [4, 8],
            "qkv_dim": [32, 64],
            "transformer_block": True,
        },
    ),
    (),
]
layers_after_pool = [
    ({"layer_type": "mlp", "dims": (32, 32)},),
    (
        {"layer_type": "mlp", "dims": (32, 32)},
        {"layer_type": "self_attention", "num_heads": 4, "qkv_dim": 12},
    ),
    (),
]


class TestConditionEncoder:
    @pytest.mark.parametrize("pooling", ["mean", "attention_token", "attention_seed"])
    @pytest.mark.parametrize("covariates_not_pooled", [[], ["pert4_skip_pool"]])
    @pytest.mark.parametrize("layers_before_pool", layers_before_pool)
    @pytest.mark.parametrize("layers_after_pool", layers_after_pool)
    @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
    @pytest.mark.parametrize("regularization", [0.0, 0.1])
    def test_condition_encoder_init(
        self, pooling, covariates_not_pooled, layers_before_pool, layers_after_pool, condition_mode, regularization
    ):
        cond_encoder = scaleflow.networks.ConditionEncoder(
            output_dim=5,
            condition_mode=condition_mode,
            regularization=regularization,
            pooling=pooling,
            covariates_not_pooled=covariates_not_pooled,
            layers_before_pool=layers_before_pool,
            layers_after_pool=layers_after_pool,
            output_dropout=0.1,
        )

        rng = jax.random.PRNGKey(111)
        encoder_rng, dropout_rng = jax.random.split(rng, 2)
        opt = optax.adam(1e-3)
        encoder_state = cond_encoder.create_train_state(encoder_rng, opt, cond)
        cond_out = encoder_state.apply_fn(
            {"params": encoder_state.params},
            cond,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        assert isinstance(cond_out, tuple)
        assert len(cond_out) == 2
        assert cond_out[0].shape == (1, 5)
        assert cond_out[1].shape == (1, 5)
