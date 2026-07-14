"""End-to-end training over the annbatch/dagloader streaming path (in-memory AnnData as source)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

import cellflow
from dagloader import SamplerConfig


def _toy_adata(n_per_combo: int = 20, drugs=("control", "d1", "d2", "d3"), lines=("A", "B")):
    rng = np.random.default_rng(0)
    rows = [(cl, dr) for cl in lines for dr in drugs for _ in range(n_per_combo)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    x = rng.normal(size=(len(obs), 5)).astype("float32")
    return ad.AnnData(X=x, obs=obs)


def _prepare_model_small(cf):
    cf.prepare_model(
        pooling="mean",
        condition_embedding_dim=8,
        time_encoder_dims=(8,),
        hidden_dims=(8,),
        decoder_dims=(8,),
    )


class TestAnnbatchTraining:
    def test_train_runs_without_split(self):
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_loaders(
            source=_toy_adata(),
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug"]},
            split_covariates=["cell_line"],
            sampler_config=SamplerConfig(batch_size=16, chunk_size=1, preload_nchunks=16),
        )
        _prepare_model_small(cf)
        cf.train(num_iterations=2, valid_freq=100)
        assert cf.solver is not None
        assert cf.dataloader is not None

    def test_train_runs_on_train_split(self):
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_loaders(
            source=_toy_adata(),
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug"]},
            split_covariates=["cell_line"],
            sampler_config=SamplerConfig(batch_size=16, chunk_size=1, preload_nchunks=16),
            split_by=["drug"],
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            split_random_state=0,
        )
        _prepare_model_small(cf)
        cf.train(num_iterations=2, valid_freq=100)
        assert cf.solver is not None
        # val/test split loaders are streamable too
        assert set(cf._annbatch_loaders) == {"train", "val", "test"}
        val_batch = next(iter(cf._annbatch_loaders["val"]))
        assert "target" in val_batch and "source" in val_batch
