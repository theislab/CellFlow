from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from cellflow.networks._utils import (
    LAYER_REGISTRY,
    BaseModule,
    MLPBlock,
    SelfAttentionBlock,
    _apply_modules,
    _get_layers,
    register_layer,
)


class DummyBlock(BaseModule):
    """A minimal custom block used to exercise the layer registry."""

    width: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        return nn.Dense(self.width)(x)


class MaskEchoBlock(BaseModule):
    """Custom block that consumes an attention mask.

    It is *not* a :class:`SelfAttentionBlock`, so ``_apply_modules`` can only
    forward the mask to it by honoring the ``takes_attention_mask`` capability
    flag (rather than an ``isinstance`` check). The block adds the sum of the
    received mask to its input so a test can assert exactly what was forwarded.
    """

    takes_attention_mask: ClassVar[bool] = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ) -> jnp.ndarray:
        total = jnp.zeros(()) if mask is None else jnp.sum(mask)
        return x + total


@pytest.fixture
def registered_blocks():
    """Register the custom blocks and restore the registry afterwards."""
    register_layer("dummy_block")(DummyBlock)
    register_layer("mask_echo")(MaskEchoBlock)
    try:
        yield
    finally:
        LAYER_REGISTRY.pop("dummy_block", None)
        LAYER_REGISTRY.pop("mask_echo", None)


class TestLayerRegistry:
    def test_builtin_blocks_registered(self):
        # decorating the existing blocks keeps the default keys resolvable
        assert LAYER_REGISTRY["mlp"] is MLPBlock
        assert LAYER_REGISTRY["self_attention"] is SelfAttentionBlock

    def test_register_and_build_custom_block(self, registered_blocks):
        assert LAYER_REGISTRY["dummy_block"] is DummyBlock

        modules = _get_layers([{"layer_type": "dummy_block", "width": 7}])

        assert len(modules) == 1
        assert isinstance(modules[0], DummyBlock)
        assert modules[0].width == 7

    def test_get_layers_resolves_builtins_and_default(self):
        # a config without ``layer_type`` still defaults to "mlp"
        modules = _get_layers(
            [
                {"dims": (8, 8)},
                {"layer_type": "self_attention", "num_heads": 2, "qkv_dim": 8},
            ]
        )
        assert isinstance(modules[0], MLPBlock)
        assert isinstance(modules[1], SelfAttentionBlock)

    def test_get_layers_appends_output_and_dropout(self):
        modules = _get_layers([{"dims": (8, 8)}], output_dim=5, dropout_rate=0.1)
        assert isinstance(modules[0], MLPBlock)
        assert isinstance(modules[1], nn.Dense)
        assert modules[1].features == 5
        assert isinstance(modules[2], nn.Dropout)

    def test_unknown_layer_type_lists_registered_keys(self):
        with pytest.raises(ValueError, match="Unknown layer type") as exc_info:
            _get_layers([{"layer_type": "does_not_exist"}])
        msg = str(exc_info.value)
        # the error should enumerate the registered keys to guide the user
        assert "mlp" in msg
        assert "self_attention" in msg

    def test_capability_flags(self):
        # the base default is False; only attention blocks flip it to True
        assert MLPBlock.takes_attention_mask is False
        assert SelfAttentionBlock.takes_attention_mask is True
        assert MaskEchoBlock.takes_attention_mask is True

    def test_apply_modules_forwards_mask_via_capability_flag(self, registered_blocks):
        class Runner(nn.Module):
            @nn.compact
            def __call__(self, x, mask, training=True):
                return _apply_modules([MaskEchoBlock()], x, mask, training)

        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((2, 4))
        mask = jnp.ones((2, 1, 3, 3))

        runner = Runner()
        variables = runner.init(rng, x, mask, training=False)
        out = runner.apply(variables, x, mask, training=False)

        # MaskEchoBlock adds sum(mask) to x, proving the mask reached a block
        # that is dispatched purely via ``takes_attention_mask`` (it is not a
        # SelfAttentionBlock).
        assert jnp.allclose(out, jnp.sum(mask))

    def test_apply_modules_non_mask_block_ignores_mask(self):
        class Runner(nn.Module):
            @nn.compact
            def __call__(self, x, mask, training=True):
                return _apply_modules([MLPBlock(dims=(4,))], x, mask, training)

        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 4))
        mask_a = jnp.ones((2, 1, 3, 3))
        mask_b = jnp.zeros((2, 1, 3, 3))

        runner = Runner()
        variables = runner.init(rng, x, mask_a, training=False)
        out_a = runner.apply(variables, x, mask_a, training=False)
        out_b = runner.apply(variables, x, mask_b, training=False)

        # a block whose capability flag is False never receives the mask, so
        # changing the mask must not change the output
        assert jnp.allclose(out_a, out_b)
