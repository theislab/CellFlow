"""Validation in the streaming path: each condition's full cell set is read via ``DAGEvalLoader``.

Validation preserves the legacy per-condition contract (``{source, condition, target}`` dicts, one entry
per condition) through :class:`~cellflow.data._annbatch.AnnbatchValidationSampler`, but reads cells via
annbatch slice reads instead of boolean-masking a materialized matrix. The regression that matters: cells
are read at the real ``sample_rep`` (e.g. an obsm key), not the internal ``"X"`` encoder-factory
placeholder. The ``val``/``test`` splits from ``split_by`` are auto-wired as evaluation sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

import cellflow
from cellflow.data._annbatch import AnnbatchValidationSampler
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


def _prepare(cf, source, *, sample_rep="X", **kwargs):
    cf.prepare_data(
        source=source,
        sample_rep=sample_rep,
        control_key="control",
        perturbation_covariates={"drug": ["drug"]},
        split_covariates=["cell_line"],
        sampler_config=_CFG,
        **kwargs,
    )


class TestAnnbatchValidation:
    def test_prepare_validation_without_setup_raises(self):
        cf = cellflow.model.CellFlowAnnbatch()
        with pytest.raises(ValueError, match="prepare_data"):
            cf.prepare_validation_data(source=_adata(), name="val")

    def test_validation_sampler_reads_matched_source_and_target(self):
        cf = cellflow.model.CellFlowAnnbatch()
        _prepare(cf, _adata())  # sample_rep="X"
        cf.prepare_validation_data(source=_adata(seed=1), name="val")
        vs = cf.validation_data["val"]
        assert isinstance(vs, AnnbatchValidationSampler)
        batch = vs.sample("on_train_end")
        assert set(batch) == {"source", "condition", "target"}
        # control-rooted: one batch per control population (2 cell lines), keyed by the drawn (line, drug)
        assert len(batch["target"]) == 2
        for k in batch["target"]:
            assert k[1] != "control"  # target is a perturbed condition
            assert np.asarray(batch["target"][k]).shape[1] == 5  # X dim
            assert np.asarray(batch["source"][k]).shape[1] == 5  # matched controls, same rep

    def test_validation_reads_obsm_sample_rep_not_x(self):
        # training streams X_pca (dim 3); validation must read X_pca, NOT X (dim 5)
        cf = cellflow.model.CellFlowAnnbatch()
        _prepare(cf, _adata(rep_dim=3), sample_rep="X_pca")
        assert cf._data_dim == 3
        cf.prepare_validation_data(source=_adata(rep_dim=3, seed=2), name="val")
        batch = cf.validation_data["val"].sample("on_train_end")
        for k in batch["target"]:
            assert np.asarray(batch["target"][k]).shape[1] == 3
            assert np.asarray(batch["source"][k]).shape[1] == 3

    def test_val_test_splits_auto_wired_as_eval_sources(self):
        cf = cellflow.model.CellFlowAnnbatch()
        _prepare(
            cf,
            _adata(),
            split_by=["drug"],
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            split_random_state=0,
        )
        # non-train splits become DAGEvalLoaders; "val" also feeds training-time validation
        assert set(cf.split_eval_loaders) == {"val", "test"}
        assert isinstance(cf.validation_data.get("val"), AnnbatchValidationSampler)

    def test_train_with_validation_runs(self):
        cf = cellflow.model.CellFlowAnnbatch()
        _prepare(cf, _adata())
        cf.prepare_validation_data(source=_adata(seed=3), name="val")
        cf.prepare_model(
            pooling="mean", condition_embedding_dim=8, time_encoder_dims=(8,), hidden_dims=(8,), decoder_dims=(8,)
        )
        cf.train(num_iterations=2, valid_freq=2)  # triggers a DAGEvalLoader validation pass
        assert cf.solver is not None
