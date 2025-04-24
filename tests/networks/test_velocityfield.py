import jax
import jax.numpy as jnp
import optax
import pytest

from cellflow.networks import _velocity_field

x_test = jnp.ones((10, 5)) * 10
t_test = jnp.ones((10, 1))
x_0_test = x_test + 2.0
cond = {"pert1": jnp.ones((1, 2, 3))}


class TestVelocityField:
    @pytest.mark.parametrize("decoder_dims", [(32, 32), (2, 2)])
    @pytest.mark.parametrize("hidden_dims", [(32, 32), (2, 2)])
    @pytest.mark.parametrize("layer_norm_before_concatenation", [True, False])
    @pytest.mark.parametrize("linear_projection_before_concatenation", [True, False])
    @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
    @pytest.mark.parametrize(
        "velocity_field_cls", [_velocity_field.ConditionalVelocityField, _velocity_field.GENOTConditionalVelocityField]
    )
    @pytest.mark.parametrize("conditioning", ["concatenation", "film", "resnet"])
    def test_velocity_field_init(
        self,
        hidden_dims,
        decoder_dims,
        layer_norm_before_concatenation,
        linear_projection_before_concatenation,
        condition_mode,
        velocity_field_cls,
        conditioning,
    ):
        linear_projection_before_concatenation = False
        vf = velocity_field_cls(
            output_dim=5,
            max_combination_length=2,
            condition_mode=condition_mode,
            condition_embedding_dim=12,
            hidden_dims=hidden_dims,
            decoder_dims=decoder_dims,
            layer_norm_before_concatenation=layer_norm_before_concatenation,
            linear_projection_before_concatenation=linear_projection_before_concatenation,
            conditioning=conditioning,
        )
        assert vf.output_dims == decoder_dims + (5,)

        vf_rng = jax.random.PRNGKey(111)
        vf_rng, apply_rng, encoder_noise_rng = jax.random.split(vf_rng, 3)
        opt = optax.adam(1e-3)
        encoder_noise = jax.random.normal(encoder_noise_rng, (x_test.shape[0], vf.condition_embedding_dim))
        vf_state = vf.create_train_state(rng=vf_rng, optimizer=opt, input_dim=5, conditions=cond)
        if isinstance(vf, _velocity_field.GENOTConditionalVelocityField):
            out, out_mean, out_logvar = vf_state.apply_fn(
                {"params": vf_state.params},
                t_test,
                x_test,
                x_0_test,
                cond,
                encoder_noise,
                train=True,
                rngs={"condition_encoder": apply_rng},
            )
        elif isinstance(vf, _velocity_field.ConditionalVelocityField):
            out, out_mean, out_logvar = vf_state.apply_fn(
                {"params": vf_state.params},
                t_test,
                x_test,
                cond,
                encoder_noise,
                train=True,
                rngs={"condition_encoder": apply_rng},
            )
        else:
            raise ValueError("Invalid velocity field class.")
        assert out.shape == (10, 5)
        assert out_mean.shape == (1, 12)
        assert out_logvar.shape == (1, 12)

        out_mean, out_logvar = vf.apply({"params": vf_state.params}, cond, method="get_condition_embedding")
        assert out_mean.shape == (1, 12)
        assert out_logvar.shape == (1, 12)
