"""Mathematical-equivalence tests for the flow-matching solvers.

These are *analytic* tests: instead of only asserting "training runs and the output
is finite", they check the exact identities the papers prescribe, so a subtle sign/
variable swap that silently degrades a model is caught deterministically on CPU.

Layers covered here:

1. Probability-path identities (Lipman'22 / Tong'23 OT-CFM straight path).
2. Flow-matching self-consistency: the regression target the loss uses must be the
   time-derivative of the *same* path the ODE integrates at inference. This is the
   invariant that distinguishes ``u_t = target - x_0`` (correct) from
   ``u_t = target - source`` (the GENOT bug).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import cellflow
from cellflow._compat import BrownianBridge, ConstantNoiseFlow
from cellflow.solvers import _genot, _otfm
from cellflow.utils import match_linear

_RNG = jax.random.PRNGKey(0)


# --------------------------------------------------------------------------------------
# 1. Probability-path identities
# --------------------------------------------------------------------------------------
class TestProbabilityPath:
    def test_straight_path_mu_and_ut(self):
        """ConstantNoiseFlow is the OT-CFM straight path: mu_t=(1-t)x0+t x1, u_t=x1-x0."""
        flow = ConstantNoiseFlow(0.0)
        x0 = jnp.array([[1.0, -2.0, 0.5]])
        x1 = jnp.array([[4.0, 1.0, -0.5]])
        for t in (0.0, 0.3, 0.7, 1.0):
            tt = jnp.array([[t]])
            mu = flow.compute_mu_t(tt, x0, x1)
            assert np.allclose(mu, (1 - t) * x0 + t * x1)
        # u_t is the (time-independent) path velocity x1 - x0.
        assert np.allclose(flow.compute_ut(jnp.array([[0.4]]), None, x0, x1), x1 - x0)

    def test_ut_is_time_derivative_of_xt(self):
        """u_t must equal d/dt x_t (σ=0). Finite-difference check for both flows."""
        for flow in (ConstantNoiseFlow(0.0), BrownianBridge(0.0)):
            x0 = jnp.array([[0.2, 1.5]])
            x1 = jnp.array([[-1.0, 3.0]])
            t = 0.35
            eps = 1e-4
            xt_plus = flow.compute_mu_t(jnp.array([[t + eps]]), x0, x1)
            xt_minus = flow.compute_mu_t(jnp.array([[t - eps]]), x0, x1)
            dxt_dt = (xt_plus - xt_minus) / (2 * eps)
            xt = flow.compute_mu_t(jnp.array([[t]]), x0, x1)
            u_t = flow.compute_ut(jnp.array([[t]]), xt, x0, x1)
            assert np.allclose(dxt_dt, u_t, atol=1e-3)


# --------------------------------------------------------------------------------------
# 2. Flow-matching self-consistency (catches the source-vs-latent bug)
# --------------------------------------------------------------------------------------
def _make_otfm_deterministic():
    vf = cellflow.networks.ConditionalVelocityField(
        output_dim=5,
        max_combination_length=2,
        condition_embedding_dim=12,
        hidden_dims=(16, 16),
        decoder_dims=(16, 16),
        condition_mode="deterministic",
    )
    return _otfm.OTFlowMatching(
        vf=vf,
        match_fn=match_linear,
        probability_path=ConstantNoiseFlow(0.0),
        optimizer=optax.adam(1e-3),
        conditions={"drug": np.random.rand(2, 1, 3)},
        rng=_RNG,
    )


def _make_genot_deterministic():
    vf = cellflow.networks.GENOTConditionalVelocityField(
        output_dim=5,
        max_combination_length=2,
        condition_embedding_dim=12,
        hidden_dims=(16, 16),
        decoder_dims=(16, 16),
        genot_source_dims=(16, 16),
        condition_mode="deterministic",
    )
    return _genot.GENOT(
        vf=vf,
        data_match_fn=match_linear,
        probability_path=ConstantNoiseFlow(0.0),
        optimizer=optax.adam(1e-3),
        source_dim=5,
        target_dim=5,
        conditions={"drug": np.random.rand(2, 1, 3)},
        rng=_RNG,
    )


def _reference_fm_loss(solver, params, rng, t, x_0, source, target, cond, enc):
    """Recompute the flow-matching loss with the paper-correct target ``target - x_0``.

    Replicates the solver's own rng discipline and velocity-field evaluation exactly,
    differing *only* in the regression target, which the papers fix to the derivative
    of the path from ``x_0`` (the ODE start), i.e. ``compute_ut(t, x_t, x_0, target)``.
    """
    rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
    x_t = solver.probability_path.compute_xt(rng_flow, t, x_0, target)
    v_t, mean_cond, logvar_cond = solver._apply_vf(
        solver.vf_state,
        params,
        t,
        x_t,
        source,
        cond,
        enc,
        {"dropout": rng_dropout, "condition_encoder": rng_encoder},
    )
    u_t = solver.probability_path.compute_ut(t, x_t, x_0, target)  # x_0, NOT source
    return jnp.mean((v_t - u_t) ** 2) + solver._encoder_loss(mean_cond, logvar_cond)


class TestFlowMatchingTarget:
    """The loss must regress onto the derivative of the path the ODE integrates."""

    n, d = 4, 5

    def _inputs(self):
        k = jax.random.split(_RNG, 4)
        source = jax.random.normal(k[0], (self.n, self.d))
        target = jax.random.normal(k[1], (self.n, self.d))
        latent = jax.random.normal(k[2], (self.n, self.d))
        t = jax.random.uniform(k[3], (self.n, 1))
        cond = {"drug": jnp.ones((1, 2, 3))}
        enc = jnp.zeros((1, 12))
        return source, target, latent, t, cond, enc

    def test_otfm_regresses_on_target_minus_source(self):
        """OTFM's path starts at the source, so its target is ``target - source``."""
        solver = _make_otfm_deterministic()
        source, target, _latent, t, cond, enc = self._inputs()
        params = solver.vf_state.params
        # OTFM: x_0 == source.
        actual = solver._flow_matching_loss(params, solver.vf_state, _RNG, t, source, source, target, cond, enc)
        expected = _reference_fm_loss(solver, params, _RNG, t, source, source, target, cond, enc)
        assert np.allclose(actual, expected, atol=1e-6)

    def test_genot_regresses_on_target_minus_latent(self):
        """GENOT's path starts at the latent, so its target must be ``target - latent``.

        This fails on the ``target - source`` bug: the ODE integrates from the latent,
        so regressing onto ``target - source`` biases every generated sample by
        ``latent - source``.
        """
        solver = _make_genot_deterministic()
        source, target, latent, t, cond, enc = self._inputs()
        params = solver.vf_state.params
        # GENOT: x_0 == latent (what predict integrates from), source is only conditioning.
        actual = solver._flow_matching_loss(params, solver.vf_state, _RNG, t, latent, source, target, cond, enc)
        expected = _reference_fm_loss(solver, params, _RNG, t, latent, source, target, cond, enc)
        assert np.allclose(actual, expected, atol=1e-6)

    def test_oracle_velocity_round_trip(self):
        """Sanity: with the correct target, integrating the constant field from the ODE
        start reaches the target; with ``target - source`` it lands at ``latent+target-source``."""
        flow = ConstantNoiseFlow(0.0)
        latent = jnp.array([[0.0, 0.0]])
        source = jnp.array([[5.0, -3.0]])
        target = jnp.array([[1.0, 2.0]])
        t = jnp.array([[0.4]])
        x_t = flow.compute_xt(_RNG, t, latent, target)
        u_correct = flow.compute_ut(t, x_t, latent, target)
        u_buggy = flow.compute_ut(t, x_t, source, target)
        assert np.allclose(latent + u_correct, target)  # correct target reaches target
        assert not np.allclose(latent + u_buggy, target)  # source-based target does not
