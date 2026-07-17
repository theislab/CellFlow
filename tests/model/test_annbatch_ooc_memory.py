"""The core cluster invariant: the perturbed target streams OUT-OF-CORE while controls live in RAM.

On Tahoe-scale data the perturbed population is the ~10^8-cell bulk that must never be materialized;
the matched controls are the small, re-drawn-every-batch population that belongs in memory. These tests
pin that split down explicitly — node identity, whether each node's backings are on-disk vs in-RAM, the
``control_in_memory`` toggle, and that the OOC/in-RAM split still matches control↔perturbed correctly —
so a regression that silently pulls the perturbed cells into RAM (an OOM on the cluster) is caught here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad
from annbatch import DatasetCollection

import cellflow
from binded import SamplerConfig
from binded._io import key_backings
from binded._loader import _is_backed

_PREP = {
    "sample_rep": "X",
    "control_key": "control",
    "perturbation_covariates": {"drug": ["drug"]},
    "split_covariates": ["cell_line"],
}
_LINE_CODE = {"A": 0.0, "B": 1.0}


def _adata(n_per_combo=30, drugs=("control", "d1", "d2"), lines=("A", "B")):
    """X encodes each cell's identity: col0 = is_control (1/0), col1 = cell-line code; rest random.

    That lets a streamed batch be checked back to its provenance (perturbed vs control, which line)
    without any obs — the loader streams X only.
    """
    rng = np.random.default_rng(0)
    rows = [(cl, dr) for cl in lines for dr in drugs for _ in range(n_per_combo)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    x = rng.normal(size=(len(obs), 5)).astype("float32")
    x[:, 0] = (obs["drug"] == "control").to_numpy().astype("float32")
    x[:, 1] = obs["cell_line"].map(_LINE_CODE).to_numpy().astype("float32")
    return ad.AnnData(X=x, obs=obs)


def _collection(tmp_path) -> DatasetCollection:
    """A grouped out-of-core collection (contiguous category runs, as the cluster data is built)."""
    h5 = tmp_path / "a.h5ad"
    _adata().write_h5ad(h5)
    dc = DatasetCollection(str(tmp_path / "c.zarr"), mode="a")
    dc.add_adatas([str(h5)], groupby=["cell_line", "drug"], shuffle=False)
    return dc


def _prepare(tmp_path, *, control_in_memory=True, **extra):
    cf = cellflow.model.CellFlowAnnbatch()
    cf.prepare_data(
        data=_collection(tmp_path),
        sampler_config=SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8),
        control_in_memory=control_in_memory,
        **_PREP,
        **extra,
    )
    return cf


class TestOutOfCoreMemorySplit:
    def test_pert_out_of_core_ctrl_in_memory(self, tmp_path):
        # The default: perturbed target streams from disk, controls are materialized into RAM.
        dc = _collection(tmp_path)
        cf = cellflow.model.CellFlowAnnbatch()
        cf.prepare_data(
            data=dc,
            sampler_config=SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8),
            control_in_memory=True,
            **_PREP,
        )
        ldr = cf._dataloader._loader
        pert, ctrl = ldr._nodes["pert"], ldr._nodes["ctrl"]

        # pert node: still the SAME out-of-core collection object (no copy / no materialization), and
        # every backing it streams is an on-disk backing — the whole point at 10^8 cells.
        assert cf._scheme.nodes["pert"].in_memory is False
        assert pert is dc, "perturbed source was copied/materialized — must stream the original collection"
        assert isinstance(pert, DatasetCollection)
        assert all(_is_backed(b) for b in key_backings(pert, "X"))

        # ctrl node: materialized into an in-memory AnnData, holding ONLY the control cells (the small
        # re-drawn population), with in-RAM (not backed) arrays.
        assert cf._scheme.nodes["ctrl"].in_memory is True
        assert isinstance(ctrl, ad.AnnData)
        assert not any(_is_backed(b) for b in key_backings(ctrl, "X"))
        assert ctrl.n_obs == 60, "in-memory ctrl must hold exactly the controls (A/control 30 + B/control 30)"
        # every materialized cell really is a control (col0 == 1) — no perturbed cell leaked into RAM.
        assert np.all(np.asarray(ctrl.X)[:, 0] == 1.0)

    def test_control_in_memory_false_streams_both(self, tmp_path):
        # Toggle off ⇒ controls stream out-of-core too (both nodes on-disk); nothing materialized.
        cf = _prepare(tmp_path, control_in_memory=False)
        ldr = cf._dataloader._loader
        assert cf._scheme.nodes["ctrl"].in_memory is False
        assert isinstance(ldr._nodes["ctrl"], DatasetCollection)
        assert all(_is_backed(b) for b in key_backings(ldr._nodes["ctrl"], "X"))
        assert all(_is_backed(b) for b in key_backings(ldr._nodes["pert"], "X"))

    def test_streamed_target_perturbed_inmem_source_control_matched(self, tmp_path):
        # Correctness of the split: across many batches the OOC target is always perturbed, the in-RAM
        # source is always a control, and both are the SAME cell line (the bind matches on cell_line).
        cf = _prepare(tmp_path)
        for _ in range(20):
            b = cf._dataloader.sample()
            tgt, src = np.asarray(b["tgt_cell_data"]), np.asarray(b["src_cell_data"])
            assert np.all(tgt[:, 0] == 0.0), "streamed target contained a control cell"
            assert np.all(src[:, 0] == 1.0), "in-memory source contained a perturbed cell"
            # class-coherent batch: one cell line per batch, and target/source share it (matched control).
            assert len(np.unique(tgt[:, 1])) == 1 and len(np.unique(src[:, 1])) == 1
            assert tgt[0, 1] == src[0, 1], "control drawn from a different cell line than the target"

    def test_train_out_of_core_pert_in_memory_ctrl(self, tmp_path):
        # End-to-end: the realistic cluster config trains, and the memory split still holds afterwards.
        cf = _prepare(tmp_path)
        cf.prepare_model(
            pooling="mean", condition_embedding_dim=8, time_encoder_dims=(8,), hidden_dims=(8,), decoder_dims=(8,)
        )
        cf.train(num_iterations=2, valid_freq=100)
        assert cf.solver is not None
        ldr = cf._dataloader._loader
        assert isinstance(ldr._nodes["pert"], DatasetCollection)  # target never pulled into RAM
        assert isinstance(ldr._nodes["ctrl"], ad.AnnData)  # controls stayed in RAM
