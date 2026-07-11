import warnings

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import cellflow
from cellflow._compat import ConstantNoiseFlow
from cellflow.solvers import _genot, _otfm
from cellflow.solvers._otfm import ClassifierFreeGuidance
from cellflow.utils import match_linear

src = {
    ("drug_1",): np.random.rand(10, 5),
    ("drug_2",): np.random.rand(10, 5),
}
cond = {
    ("drug_1",): {"drug": np.random.rand(1, 1, 3)},
    ("drug_2",): {"drug": np.random.rand(1, 1, 3)},
}
vf_rng = jax.random.PRNGKey(111)


class TestSolver:
    @pytest.mark.parametrize("solver_class", ["otfm", "genot"])
    def test_predict_batch(self, dataloader, solver_class):
        if solver_class == "otfm":
            vf_class = cellflow.networks.ConditionalVelocityField
        else:
            vf_class = cellflow.networks.GENOTConditionalVelocityField

        opt = optax.adam(1e-3)
        vf = vf_class(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        if solver_class == "otfm":
            solver = _otfm.OTFlowMatching(
                vf=vf,
                match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=opt,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )
        else:
            solver = _genot.GENOT(
                vf=vf,
                data_match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=opt,
                source_dim=5,
                target_dim=5,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )

        predict_kwargs = {"max_steps": 3, "throw": False}
        trainer = cellflow.training.CellFlowTrainer(solver=solver, predict_kwargs=predict_kwargs)
        trainer.train(
            dataloader=dataloader,
            num_iterations=2,
            valid_freq=1,
        )
        x_pred_batched = solver.predict(src, cond)

        x_pred_nonbatched = jax.tree.map(
            solver.predict,
            src,
            cond,  # type: ignore[attr-defined]
        )

        assert x_pred_batched[("drug_1",)].shape == x_pred_nonbatched[("drug_1",)].shape
        assert np.allclose(
            x_pred_batched[("drug_1",)],
            x_pred_nonbatched[("drug_1",)],
            atol=1e-1,
            rtol=1e-2,
        )

    @pytest.mark.parametrize("solver_class", ["otfm", "genot"])
    def test_predict_batched_deprecation_warning(self, dataloader, solver_class):
        if solver_class == "otfm":
            vf_class = cellflow.networks.ConditionalVelocityField
        else:
            vf_class = cellflow.networks.GENOTConditionalVelocityField

        opt = optax.adam(1e-3)
        vf = vf_class(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        if solver_class == "otfm":
            solver = _otfm.OTFlowMatching(
                vf=vf,
                match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=opt,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )
        else:
            solver = _genot.GENOT(
                vf=vf,
                data_match_fn=match_linear,
                probability_path=ConstantNoiseFlow(0.0),
                optimizer=opt,
                source_dim=5,
                target_dim=5,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )

        predict_kwargs = {"max_steps": 3, "throw": False}
        trainer = cellflow.training.CellFlowTrainer(solver=solver, predict_kwargs=predict_kwargs)
        trainer.train(
            dataloader=dataloader,
            num_iterations=2,
            valid_freq=1,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.predict(src, cond, batched=True)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "batched" in str(w[0].message)

    @pytest.mark.parametrize("ema", [0.5, 1.0])
    def test_EMA(self, dataloader, ema):
        vf_class = cellflow.networks.ConditionalVelocityField
        drug = np.random.rand(2, 1, 3)
        opt = optax.adam(1e-3)
        vf1 = vf_class(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(6, 6),
            decoder_dims=(5, 5),
        )

        solver1 = _otfm.OTFlowMatching(
            vf=vf1,
            match_fn=match_linear,
            probability_path=ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions={"drug": drug},
            rng=vf_rng,
            ema=ema,
        )
        trainer1 = cellflow.training.CellFlowTrainer(solver=solver1)
        trainer1.train(
            dataloader=dataloader,
            num_iterations=5,
            valid_freq=10,
        )

        if ema == 1.0:
            assert jax.tree.all(
                jax.tree.map(
                    lambda x, y: np.allclose(x, y, atol=1e-5, rtol=1e-2),
                    solver1.vf_state_inference.params,
                    solver1.vf_state.params,
                )
            )
        else:
            assert not jax.tree.all(
                jax.tree.map(
                    lambda x, y: np.allclose(x, y, atol=1e-5),
                    solver1.vf_state_inference.params,
                    solver1.vf_state.params,
                )
            )


def _make_otfm(condition_dropout_prob=0.0, guidance=None, condition_null="zero_embedding"):
    """Build a small OTFlowMatching solver for guidance tests."""
    vf = cellflow.networks.ConditionalVelocityField(
        output_dim=5,
        max_combination_length=2,
        condition_embedding_dim=12,
        hidden_dims=(32, 32),
        decoder_dims=(32, 32),
        condition_dropout_prob=condition_dropout_prob,
        condition_null=condition_null,
    )
    return _otfm.OTFlowMatching(
        vf=vf,
        match_fn=match_linear,
        probability_path=ConstantNoiseFlow(0.0),
        optimizer=optax.adam(1e-3),
        conditions={"drug": np.random.rand(2, 1, 3)},
        rng=vf_rng,
        guidance=guidance,
    )


def _make_genot(condition_dropout_prob=0.0, guidance=None, condition_null="zero_embedding"):
    """Build a small GENOT solver for guidance tests."""
    vf = cellflow.networks.GENOTConditionalVelocityField(
        output_dim=5,
        max_combination_length=2,
        condition_embedding_dim=12,
        hidden_dims=(32, 32),
        decoder_dims=(32, 32),
        genot_source_dims=(32, 32),
        condition_dropout_prob=condition_dropout_prob,
        condition_null=condition_null,
    )
    return _genot.GENOT(
        vf=vf,
        data_match_fn=match_linear,
        probability_path=ConstantNoiseFlow(0.0),
        optimizer=optax.adam(1e-3),
        source_dim=5,
        target_dim=5,
        conditions={"drug": np.random.rand(2, 1, 3)},
        rng=vf_rng,
        guidance=guidance,
    )


class TestGuidance:
    def test_predict_guidance_none_matches_conditional_solve(self):
        """With ``guidance=None`` predict reproduces the pre-change conditional-only solve.

        The base velocity closure is reconstructed exactly as it existed before the
        guidance seam and integrated with the same diffrax defaults; the result must
        match ``solver.predict`` bit-for-bit (no unconditional velocity is computed).
        """
        solver = _make_otfm(guidance=None)
        assert solver.guidance is None

        x = jnp.ones((4, 5))
        condition = {"drug": jnp.ones((1, 2, 3))}

        pred = solver.predict(x, condition)

        # Manually rebuild the conditional-only predict path (pre-change behavior).
        kwargs = {
            "dt0": None,
            "solver": diffrax.Tsit5(),
            "stepsize_controller": diffrax.PIDController(rtol=1e-5, atol=1e-5),
        }
        encoder_noise = jnp.zeros((1, solver.vf.condition_embedding_dim))
        params = solver.vf_state_inference.params

        def vf(t, y, args):
            p, cond_, enc = args
            return solver.vf_state_inference.apply_fn({"params": p}, t, y, cond_, enc, train=False)[0]

        def solve_ode(p, y, cond_, enc):
            term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(term, t0=0.0, t1=1.0, y0=y, args=(p, cond_, enc), **kwargs)
            return result.ys[0]

        manual = jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, None, None]))(params, x, condition, encoder_noise)

        assert np.allclose(pred, np.asarray(manual), atol=1e-6, rtol=1e-6)

    def test_classifier_free_guidance_wraps_velocity(self):
        """The wrapped velocity equals ``v_null + scale * (v_cond - v_null)``."""
        w = 3.0
        solver = _make_otfm(condition_dropout_prob=0.5, guidance=ClassifierFreeGuidance(w))
        assert isinstance(solver.guidance, ClassifierFreeGuidance)

        t = jnp.array(0.3)
        x = jnp.ones((5,))
        condition = {"drug": jnp.ones((1, 2, 3))}
        params = solver.vf_state_inference.params
        encoder_noise = jnp.zeros((1, solver.vf.condition_embedding_dim))
        args = (params, condition, encoder_noise)

        base_vf = solver._base_velocity()
        v_cond = base_vf(t, x, args)
        v_null = solver.vf_state_inference.apply_fn(
            {"params": params}, t, x, condition, encoder_noise, train=False, force_uncond=True
        )[0]

        guided_vf = solver.guidance.wrap(base_vf, solver._null_velocity())
        v_guided = guided_vf(t, x, args)

        expected = v_null + w * (v_cond - v_null)
        assert np.allclose(np.asarray(v_guided), np.asarray(expected), atol=1e-6)
        # The unconditional velocity must actually differ, otherwise the test is vacuous.
        assert not np.allclose(np.asarray(v_null), np.asarray(v_cond), atol=1e-5)

    def test_classifier_free_guidance_scale_one_is_conditional(self):
        """At ``scale=1.0`` the guided velocity reduces to the conditional velocity."""
        solver = _make_otfm(condition_dropout_prob=0.5, guidance=ClassifierFreeGuidance(1.0))

        t = jnp.array(0.7)
        x = jnp.ones((5,))
        condition = {"drug": jnp.ones((1, 2, 3))}
        params = solver.vf_state_inference.params
        encoder_noise = jnp.zeros((1, solver.vf.condition_embedding_dim))
        args = (params, condition, encoder_noise)

        base_vf = solver._base_velocity()
        v_cond = base_vf(t, x, args)
        v_guided = solver.guidance.wrap(base_vf, solver._null_velocity())(t, x, args)

        assert np.allclose(np.asarray(v_guided), np.asarray(v_cond), atol=1e-6)

    def test_condition_dropout_training_runs(self, dataloader):
        """Training with ``condition_dropout_prob>0`` exercises the drop path and stays finite."""
        solver = _make_otfm(condition_dropout_prob=0.5)
        trainer = cellflow.training.CellFlowTrainer(solver=solver, predict_kwargs={"max_steps": 3, "throw": False})
        trainer.train(dataloader=dataloader, num_iterations=2, valid_freq=10)

        pred = solver.predict(jnp.ones((4, 5)), {"drug": jnp.ones((1, 2, 3))})
        assert pred.shape == (4, 5)
        assert np.all(np.isfinite(pred))

    def test_from_ode_weight_parameterization(self):
        """``from_ode_weight(w)`` maps to ``scale = 1 + w`` and rejects negative weights."""
        assert ClassifierFreeGuidance.from_ode_weight(0.0).scale == 1.0
        assert ClassifierFreeGuidance.from_ode_weight(2.0).scale == 3.0
        with pytest.raises(ValueError, match="cfg_ode_weight must be non-negative"):
            ClassifierFreeGuidance.from_ode_weight(-1.0)

    @pytest.mark.parametrize("pooling", ["mean", "attention_token", "attention_seed"])
    def test_mask_value_null_matches_mask_filled_condition(self, pooling):
        """With ``condition_null='mask_value'``, ``force_uncond`` equals evaluating the vf on a mask-filled condition."""
        vf = cellflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            condition_dropout_prob=0.5,
            condition_null="mask_value",
            pooling=pooling,
        )
        solver = _otfm.OTFlowMatching(
            vf=vf,
            match_fn=match_linear,
            probability_path=ConstantNoiseFlow(0.0),
            optimizer=optax.adam(1e-3),
            conditions={"drug": np.random.rand(2, 1, 3)},
            rng=vf_rng,
        )

        t = jnp.array(0.3)
        x = jnp.ones((5,))
        condition = {"drug": jnp.ones((1, 2, 3))}
        params = solver.vf_state_inference.params
        encoder_noise = jnp.zeros((1, solver.vf.condition_embedding_dim))

        v_null_force = solver.vf_state_inference.apply_fn(
            {"params": params}, t, x, condition, encoder_noise, train=False, force_uncond=True
        )[0]
        # Reference null: the raw condition filled with mask_value, run through the vf.
        null_cond = {k: jnp.full_like(v, solver.vf.mask_value) for k, v in condition.items()}
        v_null_ref = solver.vf_state_inference.apply_fn(
            {"params": params}, t, x, null_cond, encoder_noise, train=False
        )[0]

        assert np.all(np.isfinite(np.asarray(v_null_force)))
        assert np.allclose(np.asarray(v_null_force), np.asarray(v_null_ref), atol=1e-6)

    def test_mask_value_null_differs_from_zero_embedding(self):
        """The two null conventions produce different unconditional velocities."""
        t = jnp.array(0.3)
        x = jnp.ones((5,))
        condition = {"drug": jnp.ones((1, 2, 3))}

        def v_null(condition_null):
            solver = _make_otfm(condition_dropout_prob=0.5, condition_null=condition_null)
            params = solver.vf_state_inference.params
            encoder_noise = jnp.zeros((1, solver.vf.condition_embedding_dim))
            return solver.vf_state_inference.apply_fn(
                {"params": params}, t, x, condition, encoder_noise, train=False, force_uncond=True
            )[0]

        assert not np.allclose(np.asarray(v_null("zero_embedding")), np.asarray(v_null("mask_value")), atol=1e-5)

    def test_theislab_parity_guidance_formula(self):
        """mask_value null + from_ode_weight reproduces theislab's ``(1+w)*v_cond - w*v_null``."""
        w = 2.0
        solver = _make_otfm(
            condition_dropout_prob=0.5,
            condition_null="mask_value",
            guidance=ClassifierFreeGuidance.from_ode_weight(w),
        )

        t = jnp.array(0.4)
        x = jnp.ones((5,))
        condition = {"drug": jnp.ones((1, 2, 3))}
        params = solver.vf_state_inference.params
        encoder_noise = jnp.zeros((1, solver.vf.condition_embedding_dim))
        args = (params, condition, encoder_noise)

        base_vf = solver._base_velocity()
        v_cond = base_vf(t, x, args)
        v_null = solver.vf_state_inference.apply_fn(
            {"params": params}, t, x, condition, encoder_noise, train=False, force_uncond=True
        )[0]

        v_guided = solver.guidance.wrap(base_vf, solver._null_velocity())(t, x, args)
        expected = (1.0 + w) * v_cond - w * v_null
        assert np.allclose(np.asarray(v_guided), np.asarray(expected), atol=1e-6)

    def test_mask_value_training_runs(self, dataloader):
        """Training with ``condition_null='mask_value'`` and dropout>0 stays finite."""
        solver = _make_otfm(condition_dropout_prob=0.5, condition_null="mask_value")
        trainer = cellflow.training.CellFlowTrainer(solver=solver, predict_kwargs={"max_steps": 3, "throw": False})
        trainer.train(dataloader=dataloader, num_iterations=2, valid_freq=10)

        pred = solver.predict(jnp.ones((4, 5)), {"drug": jnp.ones((1, 2, 3))})
        assert pred.shape == (4, 5)
        assert np.all(np.isfinite(pred))

    @pytest.mark.parametrize("condition_null", ["zero_embedding", "mask_value"])
    def test_predict_with_guidance_end_to_end(self, condition_null):
        """The full jit+vmap+diffrax predict path applies guidance (array and dict inputs)."""
        scale = 3.0
        solver = _make_otfm(
            condition_dropout_prob=0.5, condition_null=condition_null, guidance=ClassifierFreeGuidance(scale)
        )
        x = jnp.ones((6, 5))
        condition = {"drug": jnp.ones((1, 2, 3))}

        pred = solver.predict(x, condition)
        assert pred.shape == (6, 5)
        assert np.all(np.isfinite(pred))

        # Dict / batched input path.
        pred_batched = solver.predict({("a",): np.asarray(x)}, {("a",): {"drug": np.asarray(condition["drug"])}})
        assert pred_batched[("a",)].shape == (6, 5)
        assert np.all(np.isfinite(pred_batched[("a",)]))

        # The jitted predict must actually apply guidance: match a manual guided ODE solve.
        kwargs = {
            "dt0": None,
            "solver": diffrax.Tsit5(),
            "stepsize_controller": diffrax.PIDController(rtol=1e-5, atol=1e-5),
        }
        params = solver.vf_state_inference.params
        encoder_noise = jnp.zeros((1, solver.vf.condition_embedding_dim))

        def guided(t, y, args):
            p, c, e = args
            v_cond = solver.vf_state_inference.apply_fn({"params": p}, t, y, c, e, train=False)[0]
            v_null = solver.vf_state_inference.apply_fn({"params": p}, t, y, c, e, train=False, force_uncond=True)[0]
            return v_null + scale * (v_cond - v_null)

        def solve_ode(p, y, c, e):
            return diffrax.diffeqsolve(diffrax.ODETerm(guided), t0=0.0, t1=1.0, y0=y, args=(p, c, e), **kwargs).ys[0]

        manual = jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, None, None]))(params, x, condition, encoder_noise)
        assert np.allclose(pred, np.asarray(manual), atol=1e-5)

    def test_predict_guidance_transparent_and_effective_end_to_end(self):
        """End-to-end: ``scale=1`` reproduces no-guidance; stronger guidance changes the prediction."""
        x = jnp.ones((6, 5))
        condition = {"drug": jnp.ones((1, 2, 3))}

        pred_none = _make_otfm(condition_dropout_prob=0.5, condition_null="mask_value").predict(x, condition)
        pred_one = _make_otfm(
            condition_dropout_prob=0.5, condition_null="mask_value", guidance=ClassifierFreeGuidance(1.0)
        ).predict(x, condition)
        pred_strong = _make_otfm(
            condition_dropout_prob=0.5, condition_null="mask_value", guidance=ClassifierFreeGuidance(5.0)
        ).predict(x, condition)

        # scale=1 reduces to the conditional velocity -> identical to no guidance (shared init params).
        assert np.allclose(pred_none, pred_one, atol=1e-4)
        # Stronger guidance actually changes the prediction.
        assert not np.allclose(pred_none, pred_strong, atol=1e-3)

    def test_cfg_enabled_reflects_condition_dropout(self):
        """``cfg_enabled`` is True iff the velocity field was trained with condition dropout."""
        assert _make_otfm(condition_dropout_prob=0.5).cfg_enabled is True
        assert _make_otfm(condition_dropout_prob=0.0).cfg_enabled is False

    def test_per_call_guidance_scale_matches_construction_time(self):
        """A per-call ``guidance_scale=w`` matches constructing with ``guidance=ClassifierFreeGuidance(w)``."""
        w = 3.0
        x = jnp.ones((6, 5))
        condition = {"drug": jnp.ones((1, 2, 3))}

        # Same vf_rng in _make_otfm -> identical init params -> the two paths are comparable.
        per_call = _make_otfm(condition_dropout_prob=0.5).predict(x, condition, guidance_scale=w)
        construction = _make_otfm(condition_dropout_prob=0.5, guidance=ClassifierFreeGuidance(w)).predict(x, condition)
        assert np.allclose(per_call, construction, atol=1e-5)

        # And it is actually guiding: differs from the plain conditional (scale 1.0).
        plain = _make_otfm(condition_dropout_prob=0.5).predict(x, condition, guidance_scale=1.0)
        assert not np.allclose(per_call, plain, atol=1e-3)

    def test_per_call_guidance_scale_ignored_without_cfg(self):
        """Without condition dropout, a per-call ``guidance_scale`` is ignored (with a warning)."""
        x = jnp.ones((6, 5))
        condition = {"drug": jnp.ones((1, 2, 3))}
        solver = _make_otfm(condition_dropout_prob=0.0)
        assert solver.cfg_enabled is False

        plain = solver.predict(x, condition)  # scale 1.0 (default)
        with pytest.warns(UserWarning, match="guidance_scale"):
            ignored = solver.predict(x, condition, guidance_scale=2.0)
        assert np.allclose(plain, ignored, atol=1e-6)


class TestGENOTGuidance:
    """Classifier-free guidance for GENOT, mirroring the OTFM guidance tests."""

    genot_rng = jax.random.PRNGKey(0)

    def test_cfg_enabled_reflects_condition_dropout(self):
        """``cfg_enabled`` is True iff the velocity field was trained with condition dropout."""
        assert _make_genot(condition_dropout_prob=0.5).cfg_enabled is True
        assert _make_genot(condition_dropout_prob=0.0).cfg_enabled is False

    def test_force_uncond_differs_from_conditional(self):
        """The GENOT velocity field's ``force_uncond`` pass differs from the conditional one."""
        solver = _make_genot(condition_dropout_prob=0.5)
        t = jnp.array(0.3)
        x = jnp.ones((5,))
        x_0 = jnp.ones((5,))
        condition = {"drug": jnp.ones((1, 2, 3))}
        params = solver.vf_state.params
        encoder_noise = jnp.zeros((1, solver.vf.condition_embedding_dim))

        v_cond = solver.vf_state.apply_fn({"params": params}, t, x, x_0, condition, encoder_noise, train=False)[0]
        v_null = solver.vf_state.apply_fn(
            {"params": params}, t, x, x_0, condition, encoder_noise, train=False, force_uncond=True
        )[0]
        assert np.all(np.isfinite(np.asarray(v_null)))
        assert not np.allclose(np.asarray(v_cond), np.asarray(v_null), atol=1e-5)

    def test_per_call_guidance_scale_matches_construction_time(self):
        """A per-call ``guidance_scale=w`` matches constructing with ``guidance=ClassifierFreeGuidance(w)``."""
        w = 3.0
        x = jnp.ones((6, 5))
        condition = {"drug": jnp.ones((1, 2, 3))}
        rng_genot = self.genot_rng

        # Same vf_rng in _make_genot -> identical init params; same rng_genot -> identical latent.
        per_call = _make_genot(condition_dropout_prob=0.5).predict(x, condition, rng_genot=rng_genot, guidance_scale=w)
        construction = _make_genot(condition_dropout_prob=0.5, guidance=ClassifierFreeGuidance(w)).predict(
            x, condition, rng_genot=rng_genot
        )
        assert np.allclose(per_call, construction, atol=1e-5)

        # And it is actually guiding: differs from the plain conditional (scale 1.0).
        plain = _make_genot(condition_dropout_prob=0.5).predict(x, condition, rng_genot=rng_genot, guidance_scale=1.0)
        assert not np.allclose(per_call, plain, atol=1e-3)

    def test_per_call_guidance_scale_ignored_without_cfg(self):
        """Without condition dropout, a per-call ``guidance_scale`` is ignored (with a warning)."""
        x = jnp.ones((6, 5))
        condition = {"drug": jnp.ones((1, 2, 3))}
        rng_genot = self.genot_rng
        solver = _make_genot(condition_dropout_prob=0.0)
        assert solver.cfg_enabled is False

        plain = solver.predict(x, condition, rng_genot=rng_genot)  # scale 1.0 (default)
        with pytest.warns(UserWarning, match="guidance_scale"):
            ignored = solver.predict(x, condition, rng_genot=rng_genot, guidance_scale=2.0)
        assert np.allclose(plain, ignored, atol=1e-6)

    def test_condition_dropout_training_runs(self, dataloader):
        """Training GENOT with ``condition_dropout_prob>0`` exercises the drop path and stays finite."""
        solver = _make_genot(condition_dropout_prob=0.5)
        trainer = cellflow.training.CellFlowTrainer(solver=solver, predict_kwargs={"max_steps": 3, "throw": False})
        trainer.train(dataloader=dataloader, num_iterations=2, valid_freq=10)

        pred = solver.predict(jnp.ones((4, 5)), {"drug": jnp.ones((1, 2, 3))})
        assert pred.shape == (4, 5)
        assert np.all(np.isfinite(pred))
