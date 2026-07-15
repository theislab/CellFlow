"""An in-memory node samples at ``chunk_size=1`` regardless of the configured ``chunk_size``.

A materialized (in-RAM) node gets no benefit from chunked contiguous reads and the run-length rule is
meaningless for it, so :class:`~dagloader.DAGLoader` forces its ``chunk_size`` to 1. That lets a matched
control child with short runs sit in memory without blocking a ``chunk_size > 1`` perturbed stream — the
case that previously raised annbatch's run-length error on the control node.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

from dagloader import Bind, DAGLoader, Node, SamplerConfig, Scheme, uniform

LINES = ("A", "B")
COLS = ("cell_line", "drug")


def _source() -> ad.AnnData:
    # sorted by (cell_line, drug): perturbed leaves in runs of 8 (>= chunk), controls in runs of 2 (< chunk)
    rows: list[tuple[str, str]] = []
    for cl in LINES:
        rows += [(cl, "control")] * 2
        rows += [(cl, "d1")] * 8
        rows += [(cl, "d2")] * 8
    obs = pd.DataFrame(rows, columns=list(COLS))
    obs.index = obs.index.astype(str)
    x = np.random.default_rng(0).random((len(obs), 4), dtype="float32")
    return ad.AnnData(X=x, obs=obs)


def _scheme(source: ad.AnnData, *, ctrl_in_memory: bool) -> Scheme:
    pert = uniform([(cl, dr) for cl in LINES for dr in ("d1", "d2")])
    ctrl = uniform([(cl, "control") for cl in LINES])
    return Scheme(
        sources={"data": source},
        nodes={
            "pert": Node("data", COLS, "X", pert),
            "ctrl": Node("data", COLS, "X", ctrl, in_memory=ctrl_in_memory),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=("cell_line",)),),
        seed=0,
    )


def test_in_memory_node_forced_to_chunk_one():
    source = _source()
    cfg = SamplerConfig(batch_size=8, chunk_size=4, preload_nchunks=2)
    # control runs are length 2 < chunk_size 4 — streamed this raises, in memory it must build fine.
    loader = DAGLoader(_scheme(source, ctrl_in_memory=True), cfg)
    assert loader._cfg["pert"].chunk_size == 4  # streamed node keeps its chunk size
    assert loader._cfg["ctrl"].chunk_size == 1  # in-memory node forced to per-row reads
    ccfg = loader._cfg["ctrl"]
    assert ccfg.preload_nchunks > 0 and ccfg.preload_nchunks % ccfg.batch_size == 0  # valid at chunk 1
    assert isinstance(loader._nodes["ctrl"], ad.AnnData)  # materialized into RAM


def test_streamed_short_run_still_raises():
    # same short control runs, but NOT in memory → annbatch enforces the run-length rule itself.
    source = _source()
    cfg = SamplerConfig(batch_size=8, chunk_size=4, preload_nchunks=2)
    with pytest.raises(ValueError, match="run|chunk"):
        DAGLoader(_scheme(source, ctrl_in_memory=False), cfg)
