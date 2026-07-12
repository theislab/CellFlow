"""Validation in the annbatch path: a held-out in-memory AnnData, read at the real ``sample_rep``.

Validation is inherently in-memory (metrics need materialized cells), so `prepare_validation_data`
takes an AnnData in the streaming path too. The regression that matters: cells must be read at the
actual ``sample_rep`` (e.g. an obsm key), not the internal ``"X"`` placeholder the encoder factory used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

import cellflow
from dagloader import SamplerConfig

_CFG = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8)


def _adata(*, rep_dim=None, x_dim=5, n_per_combo=8, drugs=("control", "d1", "d2", "d3"), lines=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    rows = [(cl, dr) for cl in lines for dr in drugs for _ in range(n_per_combo)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=rng.normal(size=(len(obs), x_dim)).astype("float32"), obs=obs)
    if rep_dim is not None:
        adata.obsm["X_pca"] = rng.normal(size=(len(obs), rep_dim)).astype("float32")
    return adata


def _prepare(cf, source, *, sample_rep="X"):
    cf.prepare_annbatch_data(
        source=source,
        sample_rep=sample_rep,
        control_key="control",
        perturbation_covariates={"drug": ["drug"]},
        split_covariates=["cell_line"],
        sampler_config=_CFG,
    )


class TestAnnbatchValidation:
    def test_prepare_validation_without_setup_raises(self):
        cf = cellflow.model.CellFlow()
        with pytest.raises(ValueError, match="prepare_data.*prepare_annbatch_data|not initialized"):
            cf.prepare_validation_data(adata=_adata(), name="val")

    def test_validation_prepares_in_annbatch_path(self):
        cf = cellflow.model.CellFlow()
        _prepare(cf, _adata())  # sample_rep="X"
        cf.prepare_validation_data(adata=_adata(seed=1), name="val")
        vd = cf.validation_data["val"]
        assert vd is not None
        assert vd.cell_data.shape[1] == 5  # X dim

    def test_validation_reads_obsm_sample_rep_not_x(self):
        # source: obsm X_pca (dim 3) + X (dim 5); training streams X_pca → model dim 3
        cf = cellflow.model.CellFlow()
        _prepare(cf, _adata(rep_dim=3), sample_rep="X_pca")
        assert cf._data_dim == 3
        # validation must read X_pca (3), NOT X (5) — this is the fix under test
        cf.prepare_validation_data(adata=_adata(rep_dim=3, seed=2), name="val")
        assert cf.validation_data["val"].cell_data.shape[1] == 3

    def test_train_with_validation_runs(self):
        cf = cellflow.model.CellFlow()
        _prepare(cf, _adata())
        cf.prepare_validation_data(adata=_adata(seed=3), name="val")
        cf.prepare_model(
            pooling="mean", condition_embedding_dim=8, time_encoder_dims=(8,), hidden_dims=(8,), decoder_dims=(8,)
        )
        cf.train(num_iterations=2, valid_freq=2)  # triggers a validation pass
        assert cf.solver is not None
