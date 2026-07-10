"""Regression coverage for the name-based solver/VF registry refactor.

Pins that resolving solvers by name + the per-class hooks (`_match_kwargs`, `_normalize_vf_kwargs`)
reproduce the behavior of the old hardcoded `if self._solver_class == ...` chain, end to end:
class resolution, constructor-kwarg wiring, `vf_kwargs` validation, full train/predict for the
built-ins, and a freshly registered custom solver going through the whole pipeline.
"""

import inspect

import numpy as np
import pytest

import cellflow
from cellflow.networks import ConditionalVelocityField, GENOTConditionalVelocityField
from cellflow.solvers import GENOT, SOLVER_REGISTRY, OTFlowMatching, register_solver

# Kwargs `prepare_model` always supplies to the solver constructor (the rest are solver-specific and
# come from `_match_kwargs`). `optimizer`/`conditions`/`rng` are consumed via **kwargs at runtime.
_COMMON_SOLVER_KWARGS = {"vf", "probability_path", "optimizer", "conditions", "rng"}


def _prepare_data(cf) -> None:
    cf.prepare_data(
        sample_rep="X",
        control_key="control",
        perturbation_covariates={"drug": ["drug1"]},
        perturbation_covariate_reps={"drug": "drug"},
    )


def _required_init_params(cls) -> set[str]:
    """Names of `cls.__init__` parameters that have no default (excluding self/*args/**kwargs)."""
    params = inspect.signature(cls.__init__).parameters
    return {
        name
        for name, p in params.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.default is p.empty
    }


class TestStructuralRegression:
    """Fast, no model building: guards the hook <-> constructor contract and registry integrity."""

    @pytest.mark.parametrize("solver_cls", [OTFlowMatching, GENOT])
    def test_match_kwargs_cover_required_ctor_params(self, solver_cls):
        # Everything `prepare_model` would hand the constructor must include every required arg,
        # so a future change to a ctor or to `_match_kwargs` that drops one is caught here.
        passed = _COMMON_SOLVER_KWARGS | set(solver_cls._match_kwargs(match_fn=lambda a, b: a, data_dim=7))
        missing = _required_init_params(solver_cls) - passed
        assert not missing, f"prepare_model would omit required ctor args {missing} for {solver_cls.__name__}"

    def test_match_kwargs_exact_shape(self):
        f = lambda a, b: a
        assert OTFlowMatching._match_kwargs(match_fn=f, data_dim=9) == {"match_fn": f}
        assert GENOT._match_kwargs(match_fn=f, data_dim=9) == {
            "data_match_fn": f,
            "source_dim": 9,
            "target_dim": 9,
        }

    def test_registry_mutation_isolated(self):
        builtins = {"otfm": SOLVER_REGISTRY["otfm"], "genot": SOLVER_REGISTRY["genot"]}
        name = "regression_tmp"
        assert name not in SOLVER_REGISTRY
        register_solver(name, OTFlowMatching, ConditionalVelocityField)
        try:
            assert SOLVER_REGISTRY[name] == (OTFlowMatching, ConditionalVelocityField)
            register_solver(name, GENOT, GENOTConditionalVelocityField)  # re-registering overwrites
            assert SOLVER_REGISTRY[name] == (GENOT, GENOTConditionalVelocityField)
        finally:
            SOLVER_REGISTRY.pop(name, None)
        assert name not in SOLVER_REGISTRY
        # Built-ins are untouched by registering/removing other names.
        assert SOLVER_REGISTRY["otfm"] == builtins["otfm"]
        assert SOLVER_REGISTRY["genot"] == builtins["genot"]


@pytest.mark.slow
class TestBuiltinPipelineRegression:
    """End-to-end parity for the built-in solvers resolved through the registry (uses cf_trained)."""

    def test_resolved_classes_and_instances(self, cf_trained):
        cf, _ = cf_trained
        name = "genot" if isinstance(cf.solver, GENOT) else "otfm"
        solver_cls, vf_cls = SOLVER_REGISTRY[name]
        assert cf._solver_class is solver_cls
        assert cf._vf_class is vf_cls
        assert isinstance(cf.solver, solver_cls)
        assert isinstance(cf.vf, vf_cls)

    def test_solver_and_vf_construction_kwargs(self, cf_trained):
        cf, _ = cf_trained
        if isinstance(cf.solver, GENOT):
            # `data_match_fn` naming + explicit source/target dims flowed through `_match_kwargs`,
            assert cf.solver.source_dim == cf._data_dim
            assert hasattr(cf.solver, "data_match_fn")
            assert hasattr(cf.solver, "latent_noise_fn")
            # and the genot `vf_kwargs` were forwarded to the velocity field.
            assert tuple(cf.vf.genot_source_dims) == (2, 2)
        else:
            assert hasattr(cf.solver, "match_fn")
            assert not hasattr(cf.solver, "source_dim")
            assert not hasattr(cf.vf, "genot_source_dims")

    def test_predict_end_to_end(self, cf_trained):
        cf, adata = cf_trained
        assert cf.solver.is_trained
        adata_pred = adata.copy()
        adata_pred.obs["control"] = True
        pred = cf.predict(
            adata_pred,
            sample_rep="X",
            covariate_data=adata_pred.obs.iloc[:3],
            max_steps=3,
            throw=False,
        )
        assert isinstance(pred, dict)
        out = next(iter(pred.values()))
        assert out.shape[1] == cf._data_dim
        assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.slow
class TestVfKwargsEndToEnd:
    """The `vf_kwargs` validation now owned by the VF classes, exercised through `prepare_model`."""

    def test_otfm_rejects_vf_kwargs(self, adata_perturbation):
        cf = cellflow.model.CellFlow(adata_perturbation, solver="otfm")
        _prepare_data(cf)
        with pytest.raises(ValueError, match=r"`vf_kwargs` must be `None`"):
            cf.prepare_model(
                condition_embedding_dim=2,
                hidden_dims=(2, 2),
                decoder_dims=(2, 2),
                vf_kwargs={"genot_source_dims": (2, 2), "genot_source_dropout": 0.1},
            )

    def test_genot_missing_required_vf_kwarg_raises(self, adata_perturbation):
        cf = cellflow.model.CellFlow(adata_perturbation, solver="genot")
        _prepare_data(cf)
        with pytest.raises(AssertionError):
            cf.prepare_model(
                condition_embedding_dim=2,
                hidden_dims=(2, 2),
                decoder_dims=(2, 2),
                vf_kwargs={"genot_source_dropout": 0.1},  # missing `genot_source_dims`
            )


@pytest.fixture
def registered_custom_solver():
    # A working custom solver = thin OTFlowMatching subclass; inherits `_match_kwargs`. Registered
    # under a fresh name and removed afterwards so the global registry is not polluted.
    class RegistryRegressionSolver(OTFlowMatching):
        pass

    name = "regression_custom"
    assert name not in SOLVER_REGISTRY
    register_solver(name, RegistryRegressionSolver, ConditionalVelocityField)
    try:
        yield name, RegistryRegressionSolver
    finally:
        SOLVER_REGISTRY.pop(name, None)


@pytest.mark.slow
class TestCustomSolverEndToEnd:
    """A newly registered solver is not just selectable but trains and predicts through the pipeline."""

    def test_registered_custom_solver_full_pipeline(self, adata_perturbation, registered_custom_solver):
        name, solver_cls = registered_custom_solver

        cf = cellflow.model.CellFlow(adata_perturbation, solver=name)
        assert cf._solver_class is solver_cls
        _prepare_data(cf)
        cf.prepare_model(condition_embedding_dim=2, hidden_dims=(2, 2), decoder_dims=(2, 2))
        assert isinstance(cf.solver, solver_cls)

        cf.train(num_iterations=3)
        assert cf.solver.is_trained

        adata_pred = adata_perturbation.copy()
        adata_pred.obs["control"] = True
        pred = cf.predict(
            adata_pred,
            sample_rep="X",
            covariate_data=adata_pred.obs.iloc[:3],
            max_steps=3,
            throw=False,
        )
        assert isinstance(pred, dict)
        out = next(iter(pred.values()))
        assert out.shape[1] == cf._data_dim
        assert np.all(np.isfinite(np.asarray(out)))
