"""DAGLoader is picklable mid-stream: the live annbatch iterators are dropped, RNG/state is kept."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

from dagloader import Bind, DAGLoader, Node, SamplerConfig, Scheme, uniform


def _loader(seed=0):
    rows = [(cl, dr) for cl in ("A", "B") for dr in ("control", "d1", "d2") for _ in range(16)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs.index = obs.index.astype(str)
    obs["control"] = obs["drug"] == "control"
    adata = ad.AnnData(X=np.random.default_rng(0).normal(size=(len(obs), 4)).astype("float32"), obs=obs)
    cols = ("cell_line", "drug")
    combos = {tuple(r) for r in obs[list(cols)].to_numpy()}
    pert = [c for c in combos if c[1] != "control"]
    ctrl = [c for c in combos if c[1] == "control"]
    scheme = Scheme(
        sources={"data": adata},
        nodes={"pert": Node("data", cols, "X", uniform(pert)), "ctrl": Node("data", cols, "X", uniform(ctrl))},
        root="pert",
        binds=(Bind("pert", "ctrl", common=("cell_line",)),),
        seed=seed,
    )
    # condition_fn omitted so the loader is plain-picklable (no closure); state/RNG is what we test here
    return DAGLoader(scheme, SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))


def test_pickle_mid_stream_and_resume_deterministic():
    loader = _loader()
    it = iter(loader)
    for _ in range(2):  # advance into a pass → live annbatch generators + a wrapped sampler RNG exist
        next(it)
    blob = pickle.dumps(loader)  # would raise "cannot pickle 'generator'" without __getstate__/__reduce__

    # two loaders restored from the same checkpoint must produce identical streams
    la, lb = pickle.loads(blob), pickle.loads(blob)
    a = [next(la)["target"] for _ in range(4)]
    b = [next(lb)["target"] for _ in range(4)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert a[0].shape == (8, 4)
