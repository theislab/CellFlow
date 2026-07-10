import anndata as ad
import numpy as np
import pytest

import cellflow
from cellflow.networks import ConditionalVelocityField, GENOTConditionalVelocityField
from cellflow.solvers import GENOT, SOLVER_REGISTRY, OTFlowMatching, register_solver


@pytest.fixture
def dummy_adata() -> ad.AnnData:
    # `CellFlow.__init__` only stores `adata` and resolves the solver/VF classes from the registry,
    # so a minimal AnnData is enough to exercise name-based resolution.
    return ad.AnnData(X=np.zeros((3, 4), dtype=np.float32))


class TestSolverRegistry:
    def test_builtins_registered(self):
        assert SOLVER_REGISTRY["otfm"] == (OTFlowMatching, ConditionalVelocityField)
        assert SOLVER_REGISTRY["genot"] == (GENOT, GENOTConditionalVelocityField)

    @pytest.mark.parametrize(
        ("name", "solver_cls", "vf_cls"),
        [
            ("otfm", OTFlowMatching, ConditionalVelocityField),
            ("genot", GENOT, GENOTConditionalVelocityField),
        ],
    )
    def test_cellflow_resolves_from_registry(self, dummy_adata, name, solver_cls, vf_cls):
        cf = cellflow.model.CellFlow(dummy_adata, solver=name)
        assert cf._solver_class is solver_cls
        assert cf._vf_class is vf_cls

    def test_unknown_solver_raises_listing_names(self, dummy_adata):
        with pytest.raises(ValueError, match=r"Unknown solver 'nope'") as excinfo:
            cellflow.model.CellFlow(dummy_adata, solver="nope")
        # The error lists the registered names so the user knows the valid options.
        message = str(excinfo.value)
        assert "otfm" in message
        assert "genot" in message

    def test_register_custom_solver_selectable(self, dummy_adata):
        # A registered solver/VF pair implements the two integration hooks `prepare_model` reads:
        # the solver's `_match_kwargs` and the VF's `_normalize_vf_kwargs`.
        class DummySolver:
            @staticmethod
            def _match_kwargs(*, match_fn, data_dim):
                return {"match_fn": match_fn}

        class DummyVF:
            @staticmethod
            def _normalize_vf_kwargs(vf_kwargs):
                return vf_kwargs or {}

        assert "dummy" not in SOLVER_REGISTRY
        register_solver("dummy", DummySolver, DummyVF)
        try:
            assert SOLVER_REGISTRY["dummy"] == (DummySolver, DummyVF)
            cf = cellflow.model.CellFlow(dummy_adata, solver="dummy")
            assert cf._solver_class is DummySolver
            assert cf._vf_class is DummyVF
        finally:
            SOLVER_REGISTRY.pop("dummy", None)


class TestSolverHooks:
    """The per-solver/VF hooks `prepare_model` uses instead of hardcoded class branches."""

    def test_otfm_match_kwargs(self):
        def f(x, y):
            return x

        assert OTFlowMatching._match_kwargs(match_fn=f, data_dim=5) == {"match_fn": f}

    def test_genot_match_kwargs(self):
        def f(x, y):
            return x

        assert GENOT._match_kwargs(match_fn=f, data_dim=5) == {
            "data_match_fn": f,
            "source_dim": 5,
            "target_dim": 5,
        }

    def test_otfm_vf_kwargs_must_be_none(self):
        assert ConditionalVelocityField._normalize_vf_kwargs(None) == {}
        with pytest.raises(ValueError, match=r"`vf_kwargs` must be `None`"):
            ConditionalVelocityField._normalize_vf_kwargs({"genot_source_dims": (1, 1)})

    def test_genot_vf_kwargs_defaults_and_validation(self):
        # Defaulted when omitted ...
        defaulted = GENOTConditionalVelocityField._normalize_vf_kwargs(None)
        assert set(defaulted) == {"genot_source_dims", "genot_source_dropout"}
        # ... passed through when the required keys are present ...
        given = {"genot_source_dims": (2, 2), "genot_source_dropout": 0.1}
        assert GENOTConditionalVelocityField._normalize_vf_kwargs(given) is given
        # ... and rejected when a required key is missing.
        with pytest.raises(AssertionError):
            GENOTConditionalVelocityField._normalize_vf_kwargs({"genot_source_dropout": 0.1})
