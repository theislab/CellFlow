"""Out-of-core streaming over a real annbatch ``DatasetCollection``, incl. the ``chunk_size>1`` rule.

``chunk_size>1`` reads contiguous slices, so the collection must be **grouped** by the grouping
columns. We build collections both ways: grouped via ``add_adatas(groupby=...)`` and interleaved
(no groupby), and check chunked streaming trains on the former and errors clearly on the latter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad
from annbatch import DatasetCollection

import cellflow
from dagloader import SamplerConfig

_PREP = {
    "sample_rep": "X",
    "control_key": "control",
    "perturbation_covariates": {"drug": ["drug"]},
    "split_covariates": ["cell_line"],
}


def _adata(*, interleaved: bool, n_per_combo=40, drugs=("control", "d1", "d2", "d3"), lines=("A", "B")):
    rng = np.random.default_rng(0)
    rows = [(cl, dr) for cl in lines for dr in drugs for _ in range(n_per_combo)]
    if interleaved:
        rng.shuffle(rows)  # ungrouped input
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    x = rng.normal(size=(len(obs), 5)).astype("float32")
    return ad.AnnData(X=x, obs=obs)


def _collection(tmp_path, *, grouped: bool) -> DatasetCollection:
    h5 = tmp_path / "a.h5ad"
    _adata(interleaved=not grouped).write_h5ad(h5)
    dc = DatasetCollection(str(tmp_path / "c.zarr"), mode="a")
    if grouped:  # sort by the grouping columns on add → contiguous category runs
        dc.add_adatas([str(h5)], groupby=["cell_line", "drug"], shuffle=False)
    else:  # preserve the interleaved input order
        dc.add_adatas([str(h5)], shuffle=False)
    return dc


def _prepare_model_small(cf):
    cf.prepare_model(
        pooling="mean", condition_embedding_dim=8, time_encoder_dims=(8,), hidden_dims=(8,), decoder_dims=(8,)
    )


class TestOutOfCore:
    def test_grouped_collection_chunked_trains(self, tmp_path):
        cf = cellflow.model.CellFlow()
        cf.prepare_annbatch_data(
            source=_collection(tmp_path, grouped=True),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=16),
            **_PREP,
        )
        assert cf._data_dim == 5
        _prepare_model_small(cf)
        cf.train(num_iterations=2, valid_freq=100)
        assert cf.solver is not None

    def test_ungrouped_collection_chunked_raises(self, tmp_path):
        cf = cellflow.model.CellFlow()
        with pytest.raises(ValueError, match="grouped|groupby"):
            cf.prepare_annbatch_data(
                source=_collection(tmp_path, grouped=False),
                sampler_config=SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=16),
                **_PREP,
            )

    def test_ungrouped_collection_chunk1_ok(self, tmp_path):
        # chunk_size=1 streams per-row → no grouping needed even for an interleaved collection
        cf = cellflow.model.CellFlow()
        cf.prepare_annbatch_data(
            source=_collection(tmp_path, grouped=False),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=1, preload_nchunks=16),
            **_PREP,
        )
        assert cf._dataloader.sample()["tgt_cell_data"].shape == (16, 5)

    def test_grouped_collection_split_chunked(self, tmp_path):
        cf = cellflow.model.CellFlow()
        cf.prepare_annbatch_data(
            source=_collection(tmp_path, grouped=True),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=16),
            split_by=["drug"],
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            **_PREP,
        )
        assert set(cf._annbatch_loaders) == {"train", "val", "test"}

    def test_in_memory_unsorted_source_is_auto_grouped(self):
        # an in-memory (interleaved) AnnData source is grouped automatically → chunk_size>1 works
        cf = cellflow.model.CellFlow()
        cf.prepare_annbatch_data(
            source=_adata(interleaved=True),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=16),
            **_PREP,
        )
        assert cf._dataloader.sample()["src_cell_data"].shape == (16, 5)
