"""Condition-coherence of the BoundClassSampler-based DAGLoader — the guarantee the old suite missed.

Each cell's ``(cell_line, drug, control)`` is encoded into ``X`` so every yielded row can be decoded and
checked: the target batch is one perturbed condition, the ``condition`` vector matches that condition,
and the source batch is control cells of the *matched* context (``common=`` → same cell line). This is
what silently broke when dagloader's rng-wrapping scheduler met annbatch's refactored class draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

from dagloader import Bind, DAGLoader, Node, SamplerConfig, Scheme, uniform

LINES = ("A", "B")
DRUGS = ("control", "d1", "d2", "d3")
_LINE = {c: i for i, c in enumerate(LINES)}
_DRUG = {d: i for i, d in enumerate(DRUGS)}


def _adata(n_per_combo: int = 16) -> ad.AnnData:
    rows = [(cl, dr) for cl in LINES for dr in DRUGS for _ in range(n_per_combo)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    # X encodes the cell's identity so a batch row can be decoded: [cell_line, drug, is_control]
    x = np.stack(
        [
            obs["cell_line"].map(_LINE).to_numpy(),
            obs["drug"].map(_DRUG).to_numpy(),
            obs["control"].to_numpy().astype(float),
        ],
        axis=1,
    ).astype("float32")
    return ad.AnnData(X=x, obs=obs)


def _condition_fn(leaf: tuple) -> np.ndarray:
    return np.array([_LINE[leaf[0]], _DRUG[leaf[1]]], dtype=float)


def _scheme(adata: ad.AnnData, *, bind: Bind, seed: int = 0) -> Scheme:
    cols = ("cell_line", "drug")
    combos = {tuple(r) for r in adata.obs[list(cols)].to_numpy()}
    pert = [c for c in combos if c[1] != "control"]
    ctrl = [c for c in combos if c[1] == "control"]
    return Scheme(
        sources={"data": adata},
        nodes={"pert": Node("data", cols, "X", uniform(pert)), "ctrl": Node("data", cols, "X", uniform(ctrl))},
        root="pert",
        binds=(bind,),
        seed=seed,
    )


def test_common_bind_target_condition_and_context_coherent():
    adata = _adata()
    loader = DAGLoader(
        _scheme(adata, bind=Bind("pert", "ctrl", common=("cell_line",))),
        SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8),
        condition_fn=_condition_fn,
    )
    it = iter(loader)
    for _ in range(12):
        batch = next(it)
        tgt = np.asarray(batch["target"])
        src = np.asarray(batch["source"])
        cond = np.asarray(batch["condition"])

        # target batch is one perturbed condition
        assert len(np.unique(tgt[:, 0])) == 1, "target batch mixes cell lines"
        assert len(np.unique(tgt[:, 1])) == 1, "target batch mixes drugs"
        assert tgt[0, 2] == 0.0, "target must be perturbed (not control)"

        # condition vector matches that exact (cell_line, drug)
        assert np.all(cond[:, 0] == tgt[0, 0]) and np.all(cond[:, 1] == tgt[0, 1]), "condition ≠ target condition"

        # source batch is control cells of the SAME context (cell line)
        assert np.all(src[:, 2] == 1.0), "source must be control cells"
        assert len(np.unique(src[:, 0])) == 1 and src[0, 0] == tgt[0, 0], "source context ≠ target context"


def test_reps_are_aligned_same_cells():
    # two aligned reps of the target: X and an obsm copy of X. Same sampled rows → identical values.
    adata = _adata()
    adata.obsm["rep"] = adata.X.copy()
    cols = ("cell_line", "drug")
    combos = {tuple(r) for r in adata.obs[list(cols)].to_numpy()}
    pert = [c for c in combos if c[1] != "control"]
    ctrl = [c for c in combos if c[1] == "control"]
    scheme = Scheme(
        sources={"data": adata},
        nodes={
            "pert": Node("data", cols, ("X", "obsm/rep"), uniform(pert)),
            "ctrl": Node("data", cols, "X", uniform(ctrl)),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=("cell_line",)),),
        seed=0,
    )
    loader = DAGLoader(scheme, SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))
    batch = next(iter(loader))
    x = np.asarray(batch["target_reps"]["X"])
    rep = np.asarray(batch["target_reps"]["obsm/rep"])
    np.testing.assert_array_equal(x, rep)  # aligned reps must be the same cells
