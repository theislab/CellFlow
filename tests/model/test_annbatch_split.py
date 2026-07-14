"""Tests for the annbatch path split wiring on :class:`~cellflow.model.CellFlow`.

The Scheme + condition + loaders are built from an in-memory ``AnnData`` used as the ``source`` (the
``dagloader`` is container-agnostic), so no ``DatasetCollection`` is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")  # dagloader (and thus the annbatch path) needs annbatch

import anndata as ad

import cellflow
from dagloader import SamplerConfig, perturbation_scheme

_CFG = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8)


def _toy_adata(n_per_combo: int = 8, drugs=("control", "d1", "d2", "d3", "d4", "d5"), lines=("A", "B")):
    rng = np.random.default_rng(0)
    rows = [(cl, dr) for cl in lines for dr in drugs for _ in range(n_per_combo)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    x = rng.normal(size=(len(obs), 5)).astype("float32")
    return ad.AnnData(X=x, obs=obs)


def _toy_scheme():
    return perturbation_scheme(
        _toy_adata(n_per_combo=2),
        context=["cell_line"],
        perturbation=["drug"],
        control_values={"drug": "control"},
        key="X",
    )


class TestSplitAnnbatchData:
    """`split_annbatch_data` operates on an already-built Scheme (injected here)."""

    def test_without_scheme_raises(self):
        cf = cellflow.model.CellFlowAnnbatch()
        with pytest.raises(ValueError, match="No annbatch `Scheme`"):
            cf.split_annbatch_data(split_by=["drug"])

    def test_split_with_injected_scheme(self):
        cf = cellflow.model.CellFlowAnnbatch()
        cf._scheme = _toy_scheme()
        df = cf.split_annbatch_data(split_by=["drug"], random_state=0)
        assert list(df.columns) == ["cell_line", "drug", "split"]
        assert set(df["split"]) == {"train", "val", "test"}
        assert set(cf._split_schemes) == {"train", "val", "test"}
        for sch in cf._split_schemes.values():  # controls carried into every split
            assert sch.nodes["ctrl"].weights == cf._scheme.nodes["ctrl"].weights


class TestPrepareAnnbatchData:
    """`prepare_loaders` builds Scheme + condition + loaders from an AnnData source."""

    def _prepare(self, cf, **kwargs):
        return cf.prepare_loaders(
            source=_toy_adata(),
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug"]},
            split_covariates=["cell_line"],
            sampler_config=_CFG,
            **kwargs,
        )

    def test_requires_sampler_config(self):
        cf = cellflow.model.CellFlowAnnbatch()
        with pytest.raises(ValueError, match="sampler_config` is required"):
            cf.prepare_loaders(
                source=_toy_adata(),
                sample_rep="X",
                control_key="control",
                perturbation_covariates={"drug": ["drug"]},
            )

    def test_builds_scheme_condition_and_loader_without_split(self):
        cf = cellflow.model.CellFlowAnnbatch()
        self._prepare(cf)
        assert cf._scheme is not None and cf._scheme.root == "pert"
        assert set(cf._annbatch_loaders) == {"train"}  # no split → single "train"
        assert cf._dataloader is not None  # DAGLoaderAdapter wired for train()
        # condition embeddings assembled (drug is categorical → one-hot), data dim from X
        assert set(cf._condition_data) == {"drug"}
        assert cf._data_dim == 5
        assert cf._split_schemes is None

    def test_split_builds_per_split_loaders(self):
        cf = cellflow.model.CellFlowAnnbatch()
        assert self._prepare(cf, split_by=["drug"], split_random_state=0) is None  # prepare_* returns None
        assert set(cf._split_assignment["split"]) == {"train", "val", "test"}
        assert set(cf._annbatch_loaders) == {"train", "val", "test"}
        assert set(cf._annbatch_sampler_configs) == {"train", "val", "test"}
        assert cf._dataloader is not None

    def test_per_split_sampler_config(self):
        cf = cellflow.model.CellFlowAnnbatch()
        train = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8)
        small = SamplerConfig(batch_size=4, chunk_size=1, preload_nchunks=4)
        cf.prepare_loaders(
            source=_toy_adata(),
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug"]},
            split_covariates=["cell_line"],
            split_by=["drug"],
            sampler_config={"train": train, "val": small, "test": small},
        )
        assert cf._annbatch_sampler_configs["train"].batch_size == 8
        assert cf._annbatch_sampler_configs["val"].batch_size == 4

    def test_per_split_missing_split_raises(self):
        cf = cellflow.model.CellFlowAnnbatch()
        with pytest.raises(ValueError, match="missing config"):
            cf.prepare_loaders(
                source=_toy_adata(),
                sample_rep="X",
                control_key="control",
                perturbation_covariates={"drug": ["drug"]},
                split_covariates=["cell_line"],
                split_by=["drug"],
                sampler_config={"train": _CFG},
            )

    def test_streamed_batch_shapes(self):
        cf = cellflow.model.CellFlowAnnbatch()
        self._prepare(cf)
        batch = cf._dataloader.sample()
        assert batch["src_cell_data"].shape == (8, 5)
        assert batch["tgt_cell_data"].shape == (8, 5)
        assert batch["condition"]["drug"].shape[0] == 1  # one condition per batch, leading axis

    def test_sparse_source_batches_densified(self):
        # dagloader streams sparse for a sparse source; the model-boundary adapter densifies.
        import scipy.sparse as sp

        adata = _toy_adata()
        adata.X = sp.csr_matrix(adata.X)
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_loaders(
            source=adata,
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug"]},
            split_covariates=["cell_line"],
            sampler_config=_CFG,
        )
        batch = cf._dataloader.sample()
        assert not sp.issparse(batch["src_cell_data"]) and not sp.issparse(batch["tgt_cell_data"])
        assert batch["src_cell_data"].shape == (8, 5) and batch["tgt_cell_data"].shape == (8, 5)
