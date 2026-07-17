"""Out-of-core streaming over a real annbatch ``DatasetCollection``, incl. the ``chunk_size>1`` rule.

``chunk_size>1`` reads contiguous slices, so every contiguous run of each category must be
``>= chunk_size`` (annbatch's run-length rule; a category may span several runs). We build collections
grouped via ``add_adatas(groupby=...)``, interleaved (short runs → error), and fragmented-but-valid
(multiple long runs per category → accepted), and check chunked streaming behaves accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("annbatch")

import anndata as ad
from annbatch import DatasetCollection

import cellflow
from binded import SamplerConfig

_PREP = {
    "sample_rep": "X",
    "control_key": "control",
    "perturbation_covariates": {"drug": ["drug"]},
    "split_covariates": ["cell_line"],
}


def _adata(*, interleaved: bool, n_per_combo=40, drugs=("control", "d1", "d2", "d3"), lines=("A", "B"), sparse=False):
    rng = np.random.default_rng(0)
    rows = [(cl, dr) for cl in lines for dr in drugs for _ in range(n_per_combo)]
    if interleaved:
        rng.shuffle(rows)  # ungrouped input
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    x = rng.normal(size=(len(obs), 5)).astype("float32")
    return ad.AnnData(X=sp.csr_matrix(x) if sparse else x, obs=obs)


def _collection(tmp_path, *, grouped: bool, sparse=False) -> DatasetCollection:
    h5 = tmp_path / "a.h5ad"
    _adata(interleaved=not grouped, sparse=sparse).write_h5ad(h5)
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
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(
            data=_collection(tmp_path, grouped=True),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=16),
            **_PREP,
        )
        assert cf._data_dim == 5
        _prepare_model_small(cf)
        cf.train(num_iterations=2, valid_freq=100)
        assert cf.solver is not None

    def test_ungrouped_collection_chunked_raises(self, tmp_path):
        # annbatch enforces the run-length rule itself when building the sampler (no cellflow pre-check).
        cf = cellflow.model.CellFlowAnnbatch()
        with pytest.raises(ValueError, match="chunk_size|[Rr]e-chunk|run"):
            cf.prepare_data(
                data=_collection(tmp_path, grouped=False),
                sampler_config=SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=16),
                **_PREP,
            )

    def test_ungrouped_collection_chunk1_ok(self, tmp_path):
        # chunk_size=1 streams per-row → no grouping needed even for an interleaved collection
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(
            data=_collection(tmp_path, grouped=False),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=1, preload_nchunks=16),
            **_PREP,
        )
        assert cf._dataloader.sample()["tgt_cell_data"].shape == (16, 5)

    def test_grouped_collection_split_chunked(self, tmp_path):
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(
            data=_collection(tmp_path, grouped=True),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=16),
            split_by=["drug"],
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            **_PREP,
        )
        assert set(cf.split_eval_loaders) == {"val", "test"}  # non-train splits read via EvalLoader

    def test_in_memory_unsorted_source_is_auto_grouped(self):
        # an in-memory (interleaved) AnnData source is grouped automatically → chunk_size>1 works
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(
            data=_adata(interleaved=True),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=16),
            **_PREP,
        )
        assert cf._dataloader.sample()["src_cell_data"].shape == (16, 5)

    def test_fragmented_collection_chunked_ok(self, tmp_path):
        # each category in TWO contiguous runs (fragmented) — valid as long as every run >= chunk_size.
        # This is the case annbatch accepts but a "one run per class" check would wrongly reject.
        block = 10
        rows = [(cl, dr) for _ in range(2) for cl in ("A", "B") for dr in ("control", "d1", "d2") for _ in range(block)]
        obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
        obs["control"] = obs["drug"] == "control"
        obs.index = obs.index.astype(str)
        x = np.random.default_rng(0).normal(size=(len(obs), 5)).astype("float32")
        (tmp_path / "f.h5ad").parent.mkdir(exist_ok=True)
        ad.AnnData(X=x, obs=obs).write_h5ad(tmp_path / "f.h5ad")
        dc = DatasetCollection(str(tmp_path / "fc.zarr"), mode="a").add_adatas(
            [str(tmp_path / "f.h5ad")], shuffle=False
        )

        from binded._io import leaf_codes, obs_columns

        codes, _ = leaf_codes(obs_columns(dc, ["cell_line", "drug"]), ["cell_line", "drug"])
        n_runs = 1 + int((np.diff(codes) != 0).sum())
        assert n_runs > 6, f"expected a fragmented collection (>6 runs for 6 categories), got {n_runs}"

        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(  # chunk_size=4 <= run length 10 → must be accepted
            data=dc, sampler_config=SamplerConfig(batch_size=8, chunk_size=4, preload_nchunks=8), **_PREP
        )
        assert cf._dataloader.sample()["tgt_cell_data"].shape == (8, 5)

    def test_sparse_collection_prepares(self, tmp_path):
        # a SPARSE-X collection stores X as a zarr *group* (no .shape); key_backings must wrap it so
        # annbatch's add_datasets accepts it — regression for the "'Group' has no attribute 'shape'" error.
        # Construction (which runs add_datasets) is the assertion here; reading sparse batches needs cupy.
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(
            data=_collection(tmp_path, grouped=True, sparse=True),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=1, preload_nchunks=16),
            **_PREP,
        )
        assert cf._data_dim == 5  # Loader built (add_datasets accepted the sparse-group backing)

    def test_control_in_memory_materializes(self, tmp_path):
        # control_in_memory tells binded to materialize the ctrl node into RAM (Node.in_memory);
        # prepare_data building the loader runs materialize_node over the (sparse) collection.
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(
            data=_collection(tmp_path, grouped=True, sparse=True),
            sampler_config=SamplerConfig(batch_size=16, chunk_size=1, preload_nchunks=16),
            control_in_memory=True,
            **_PREP,
        )
        assert cf._scheme.nodes["ctrl"].in_memory is True  # cellflow flagged it
        assert isinstance(cf._dataloader._loader._nodes["ctrl"], ad.AnnData)  # binded materialized it

    def test_chunk_drops_short_run_perturbed_only(self, tmp_path):
        # A perturbed condition with a big TOTAL but a sub-chunk sliver run is dropped (per-RUN, not
        # per-total); controls are never filtered. d2 has runs [10, 2] (total 12) → its run of 2 < chunk 4
        # drops it; d1 has runs [10, 10] → kept; control [10, 10] → kept and streamed.
        def blk(dr, n):
            return [("A", dr)] * n

        rows = blk("control", 10) + blk("d1", 10) + blk("d2", 10) + blk("control", 10) + blk("d1", 10) + blk("d2", 2)
        obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
        obs["control"] = obs["drug"] == "control"
        obs.index = obs.index.astype(str)
        x = np.random.default_rng(0).normal(size=(len(obs), 5)).astype("float32")
        ad.AnnData(X=x, obs=obs).write_h5ad(tmp_path / "s.h5ad")
        dc = DatasetCollection(str(tmp_path / "sc.zarr"), mode="a").add_adatas(
            [str(tmp_path / "s.h5ad")], shuffle=False
        )

        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(data=dc, sampler_config=SamplerConfig(batch_size=8, chunk_size=4, preload_nchunks=8), **_PREP)
        pos = {leaf[1] for leaf, w in cf._scheme.nodes["pert"].weights.items() if w > 0}
        assert pos == {"d1"}  # d2 dropped for its run of 2 < chunk 4, despite total 12
        assert cf._dataloader.sample()["tgt_cell_data"].shape == (8, 5)
