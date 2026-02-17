"""Compatibility layer for ``ott-jax`` across versions.

``ott-jax>=0.6`` removed ``ott.neural.methods.flows.dynamics`` and the
``ott.neural.networks.velocity_field.VelocityField`` (flax linen) class.
This module re-exports the symbols needed by CellFlow so that both
``ott-jax>=0.5,<0.6`` and ``ott-jax>=0.6`` are supported.
"""

# ---------------------------------------------------------------------------
# Probability-path dynamics (BaseFlow, ConstantNoiseFlow, BrownianBridge)
#
# For ott-jax <0.6 we import directly from ott.  For ott-jax >=0.6 the
# module was removed, so we provide a vendored copy below.
#
# The fallback classes are a verbatim copy of
#   ott.neural.methods.flows.dynamics
# from ott-jax 0.5.0 (commit 690b1ae, 2024-12-03).
# ott-jax is licensed under the Apache License 2.0, which permits
# reproduction and distribution of derivative works provided the license
# and copyright notice are retained.  See:
#   https://github.com/ott-jax/ott/blob/0.5.0/LICENSE
# ---------------------------------------------------------------------------
try:
    from ott.neural.methods.flows.dynamics import (  # ott-jax <0.6
        BaseFlow,
        BrownianBridge,
        ConstantNoiseFlow,
    )
except ImportError:
    # -- Vendored from ott-jax 0.5.0 (Apache-2.0) --------------------------
    # Source: src/ott/neural/methods/flows/dynamics.py
    # Copyright OTT-JAX contributors
    # -----------------------------------------------------------------------
    import abc

    import jax
    import jax.numpy as jnp

    class BaseFlow(abc.ABC):
        """Base class for all flows."""

        def __init__(self, sigma: float):
            self.sigma = sigma

        @abc.abstractmethod
        def compute_mu_t(self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray: ...

        @abc.abstractmethod
        def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray: ...

        @abc.abstractmethod
        def compute_ut(self, t: jnp.ndarray, x: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray: ...

        def compute_xt(self, rng: jax.Array, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
            """Sample from the probability path."""
            noise = jax.random.normal(rng, shape=x0.shape)
            mu_t = self.compute_mu_t(t, x0, x1)
            sigma_t = self.compute_sigma_t(t)
            return mu_t + sigma_t * noise

    class _StraightFlow(BaseFlow, abc.ABC):
        def compute_mu_t(self, t, x0, x1):
            return (1.0 - t) * x0 + t * x1

        def compute_ut(self, t, x, x0, x1):
            del t, x
            return x1 - x0

    class ConstantNoiseFlow(_StraightFlow):
        r"""Flow with straight paths and constant noise :math:`\sigma`."""

        def compute_sigma_t(self, t):
            return jnp.full_like(t, fill_value=self.sigma)

    class BrownianBridge(_StraightFlow):
        r"""Brownian Bridge with :math:`\sigma_t = \sigma \sqrt{t(1-t)}`."""

        def compute_sigma_t(self, t):
            return self.sigma * jnp.sqrt(t * (1.0 - t))

        def compute_ut(self, t, x, x0, x1):
            drift_term = (1 - 2 * t) / (2 * t * (1 - t)) * (x - (t * x1 + (1 - t) * x0))
            control_term = x1 - x0
            return drift_term + control_term


__all__ = ["BaseFlow", "ConstantNoiseFlow", "BrownianBridge"]
