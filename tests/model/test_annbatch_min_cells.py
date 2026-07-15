"""Tests for the ``min_cells_per_condition`` filter on the annbatch streaming path.

The knob zero-weights perturbed conditions with too few *total* cells — a scientific filter on
untrainable tiny conditions, and the lever that unblocks ``chunk_size > 1`` (a zero-weight leaf is
exempt from annbatch's run-length rule). Built from an in-memory ``AnnData`` source (the ``dagloader``
is container-agnostic), so no ``DatasetCollection`` is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

import cellflow
from dagloader import SamplerConfig

# a couple of tiny (sub-threshold) conditions alongside normal ones; the filter drops only the tiny ones.
# four "big" drugs so a default 60/20/20 split over the kept conditions still fills every split.
_COUNTS = {"control": 6, "big1": 4, "big2": 4, "big3": 4, "big4": 4, "tiny1": 1, "tiny2": 2}
_BIG = {"big1", "big2", "big3", "big4"}
_THRESHOLD = 3  # keeps control/big* (>= 3 cells), drops tiny1 (1) and tiny2 (2)

_CFG_CHUNK1 = SamplerConfig(batch_size=4, chunk_size=1, preload_nchunks=4)
_CFG_CHUNK2 = SamplerConfig(batch_size=4, chunk_size=2, preload_nchunks=2)


def _toy_adata(counts: dict[str, int] = _COUNTS, line: str = "A") -> ad.AnnData:
    """One cell line, `counts[drug]` cells per drug; rows shuffled so the build must sort them."""
    rng = np.random.default_rng(0)
    rows = [(line, drug) for drug, n in counts.items() for _ in range(n)]
    rows = [rows[i] for i in rng.permutation(len(rows))]  # shuffled → build sorts into contiguous runs
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    x = rng.normal(size=(len(obs), 5)).astype("float32")
    return ad.AnnData(X=x, obs=obs)


def _prepare(cf, *, sampler_config, **kwargs):
    return cf.prepare_data(
        source=_toy_adata(),
        sample_rep="X",
        control_key="control",
        perturbation_covariates={"drug": ["drug"]},
        split_covariates=["cell_line"],
        sampler_config=sampler_config,
        **kwargs,
    )


def _positive_drugs(cf) -> set[str]:
    """The drugs of the perturbed root node's positive-weight leaves (cols = (cell_line, drug))."""
    return {leaf[1] for leaf, w in cf._scheme.nodes["pert"].weights.items() if w > 0}


class TestMinCellsPerCondition:
    def test_filter_drops_tiny_conditions_from_positive_leaves(self):
        # (a) with the filter, the tiny conditions are absent from the root's positive-weight leaves.
        cf = cellflow.model.CellFlowAnnbatch()
        _prepare(cf, sampler_config=_CFG_CHUNK1, min_cells_per_condition=_THRESHOLD)
        assert _positive_drugs(cf) == _BIG  # tiny1/tiny2 dropped
        # dropped leaves are retained as zero-weight (exemption), not deleted, so the source is unchanged.
        all_drugs = {leaf[1] for leaf in cf._scheme.nodes["pert"].weights}
        assert {"tiny1", "tiny2"} <= all_drugs

    def test_default_leaves_scheme_unchanged(self):
        # (c) default (0) drops nothing → every perturbed leaf keeps weight 1.0 (== uniform).
        cf = cellflow.model.CellFlowAnnbatch()
        _prepare(cf, sampler_config=_CFG_CHUNK1)  # min_cells_per_condition defaults to 0
        weights = cf._scheme.nodes["pert"].weights
        assert _positive_drugs(cf) == _BIG | {"tiny1", "tiny2"}
        assert set(weights.values()) == {1.0}  # uniform: nothing zero-weighted

    def test_default_matches_explicit_zero(self):
        # default and an explicit 0 produce identical root weights (byte-identical default path).
        cf0, cf_explicit = cellflow.model.CellFlowAnnbatch(), cellflow.model.CellFlowAnnbatch()
        _prepare(cf0, sampler_config=_CFG_CHUNK1)
        _prepare(cf_explicit, sampler_config=_CFG_CHUNK1, min_cells_per_condition=0)
        assert cf0._scheme.nodes["pert"].weights == cf_explicit._scheme.nodes["pert"].weights

    def test_chunk_gt_1_raises_without_filter(self):
        # motivation: a 1-cell condition is a run of 1 < chunk_size 2 → the strict check raises.
        cf = cellflow.model.CellFlowAnnbatch()
        with pytest.raises(ValueError, match="run of only 1|chunk_size"):
            _prepare(cf, sampler_config=_CFG_CHUNK2)

    def test_chunk_gt_1_ok_with_filter(self):
        # (b) filtering the tiny (short-run) conditions unblocks chunk_size > 1 — no raise, and the loader
        # streams: annbatch's ClassSampler exempts the zero-weight tiny leaves at iteration time too.
        cf = cellflow.model.CellFlowAnnbatch()
        _prepare(cf, sampler_config=_CFG_CHUNK2, min_cells_per_condition=_THRESHOLD)
        assert _positive_drugs(cf) == _BIG
        batch = cf._dataloader.sample()  # end-to-end chunk_size=2 read
        assert batch["tgt_cell_data"].shape == (4, 5)

    def test_filter_flows_through_split(self):
        # zero-weighted conditions never reach the split universe (split_scheme splits positive weights).
        cf = cellflow.model.CellFlowAnnbatch()
        _prepare(
            cf,
            sampler_config=_CFG_CHUNK1,
            min_cells_per_condition=_THRESHOLD,
            split_by=["drug"],
            split_random_state=0,
        )
        assert set(cf._split_assignment["drug"]) == _BIG  # no tiny drugs in any split

    def test_threshold_dropping_everything_raises(self):
        # a threshold above every condition's count zero-weights the whole root → a clear error.
        cf = cellflow.model.CellFlowAnnbatch()
        with pytest.raises(ValueError, match="dropped every perturbed condition"):
            _prepare(cf, sampler_config=_CFG_CHUNK1, min_cells_per_condition=1000)
