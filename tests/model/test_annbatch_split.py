"""Tests for :meth:`cellflow.model.CellFlow.split_annbatch_data`.

``prepare_annbatch_data`` is not implemented yet (it will build the ``Scheme`` from a
``DatasetCollection``), so these inject a scheme built from a tiny in-memory ``AnnData`` directly onto
the model — exercising the split wiring without a collection or a loader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")  # dagloader (and thus split_annbatch_data) needs annbatch

import anndata as ad

import cellflow
from dagloader import perturbation_scheme


def _toy_scheme():
    drugs = ["control", "d1", "d2", "d3", "d4", "d5"]
    rows = [(cl, dr) for cl in ("A", "B") for dr in drugs for _ in range(2)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=np.zeros((len(obs), 3), dtype="float32"), obs=obs)
    return perturbation_scheme(
        adata,
        context=["cell_line"],
        perturbation=["drug"],
        control_values={"drug": "control"},
        key="X",
    )


class TestSplitAnnbatchData:
    def test_without_scheme_raises(self):
        cf = cellflow.model.CellFlow()
        with pytest.raises(ValueError, match="No annbatch `Scheme`"):
            cf.split_annbatch_data(split_by=["drug"])

    def test_split_with_injected_scheme(self):
        cf = cellflow.model.CellFlow()
        cf._scheme = _toy_scheme()  # stand in for `prepare_annbatch_data` (still a TODO)

        df = cf.split_annbatch_data(split_by=["drug"], random_state=0)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["cell_line", "drug", "split"]
        assert set(df["split"]) == {"train", "val", "test"}
        assert cf._split_schemes is not None
        assert set(cf._split_schemes) == {"train", "val", "test"}
        # controls carried through into every split (matched-control availability)
        for sch in cf._split_schemes.values():
            assert sch.nodes["ctrl"].weights == cf._scheme.nodes["ctrl"].weights

    def test_custom_ratios_and_force(self):
        cf = cellflow.model.CellFlow()
        cf._scheme = _toy_scheme()
        df = cf.split_annbatch_data(
            split_by=["drug"],
            ratios={"train": 0.8, "test": 0.2},
            force_training_values={"drug": "d1"},
            random_state=3,
        )
        assert set(df["split"]) == {"train", "test"}
        assert df.loc[df["drug"] == "d1", "split"].eq("train").all()


class TestPrepareAnnbatchSplitStep:
    """The split step wired into `prepare_annbatch_data` (the Scheme-building part is still a TODO)."""

    def test_without_scheme_raises_not_implemented(self):
        # normal path: no pre-built scheme → the Scheme-building TODO raises
        cf = cellflow.model.CellFlow()
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            cf.prepare_annbatch_data(
                source=None,
                sample_rep="X",
                control_key="control",
                perturbation_covariates={"drug": ["drug"]},
                split_by=["drug"],
            )

    def test_split_step_runs_when_scheme_present(self):
        # inject a Scheme (stands in for the TODO Scheme-building) → the split step runs in-call
        cf = cellflow.model.CellFlow()
        cf._scheme = _toy_scheme()
        result = cf.prepare_annbatch_data(
            source=None,
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug"]},
            split_covariates=["cell_line"],
            split_by=["drug"],
            split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            split_random_state=0,
        )
        assert result is None  # prepare_* returns None, like prepare_data
        assert cf._split_schemes is not None
        assert set(cf._split_schemes) == {"train", "val", "test"}
        assert isinstance(cf._split_assignment, pd.DataFrame)
        assert set(cf._split_assignment["split"]) == {"train", "val", "test"}

    def test_no_split_when_split_by_omitted(self):
        cf = cellflow.model.CellFlow()
        cf._scheme = _toy_scheme()
        cf.prepare_annbatch_data(
            source=None,
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug"]},
        )
        assert cf._split_schemes is None
        assert cf._split_assignment is None
