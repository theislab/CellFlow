"""``DAGEvalLoader``: control-rooted eval reader (Sequential control inner + BoundClassSampler target).

For each control population (context, e.g. ``cell_line``) the source is **all** its control cells (read in
full via the Sequential inner); the target is a matched perturbed batch (annbatch samples a drug within
the context). The condition is the perturbed leaf the target drew. Cells carry a unique index in ``X`` so
we can assert the source is exactly the context's controls and the target is that context's perturbed cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

from dagloader import Bind, DAGEvalLoader, Node, SamplerConfig, Scheme, uniform

LINES = ("A", "B")
DRUGS = ("control", "d1", "d2", "d3")
_LINE = {c: i for i, c in enumerate(LINES)}
_DRUG = {d: i for i, d in enumerate(DRUGS)}
_CFG = SamplerConfig(batch_size=4, chunk_size=1, preload_nchunks=4)


def _adata(n_per_combo: int = 8) -> ad.AnnData:
    rows = [(cl, dr) for cl in LINES for dr in DRUGS for _ in range(n_per_combo)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    x = np.stack(
        [
            obs["cell_line"].map(_LINE).to_numpy(),
            obs["drug"].map(_DRUG).to_numpy(),
            obs["control"].to_numpy().astype(float),
            np.arange(len(obs)),  # unique cell index
        ],
        axis=1,
    ).astype("float32")
    return ad.AnnData(x, obs=obs)


def _condition_fn(leaf: tuple) -> dict[str, np.ndarray]:
    return {"drug": np.array([[_LINE[leaf[0]], _DRUG[leaf[1]]]], dtype=float)}


def _scheme(adata: ad.AnnData) -> Scheme:
    cols = ("cell_line", "drug")
    combos = {tuple(r) for r in adata.obs[list(cols)].to_numpy()}
    pert = [c for c in combos if c[1] != "control"]
    ctrl = [c for c in combos if c[1] == "control"]
    return Scheme(
        sources={"data": adata},
        nodes={"pert": Node("data", cols, "X", uniform(pert)), "ctrl": Node("data", cols, "X", uniform(ctrl))},
        root="pert",
        binds=(Bind("pert", "ctrl", common=("cell_line",)),),
        seed=0,
    )


def _idx(batch_x) -> set[int]:
    return set(np.asarray(batch_x)[:, 3].astype(int).tolist())


def test_source_is_all_controls_of_context_target_matches_condition():
    adata = _adata()
    obs = adata.obs
    loader = DAGEvalLoader(_scheme(adata), _CFG, _condition_fn)
    assert set(loader.control_populations) == {("A", "control"), ("B", "control")}

    seen_ctx = []
    for out in loader.iter_conditions():  # one batch per control population
        cl, dr = out["leaf"]
        seen_ctx.append(cl)
        assert dr != "control"  # target is a perturbed condition
        # source = ALL control cells of this cell line (read in full)
        src_truth = set(np.flatnonzero((obs["cell_line"] == cl).to_numpy() & obs["control"].to_numpy()).tolist())
        assert _idx(out["source"]) == src_truth
        # target cells are perturbed cells of this (cell_line, drug)
        rows = np.asarray(out["target"])
        assert np.all(rows[:, 0].astype(int) == _LINE[cl])  # same cell line
        assert np.all(rows[:, 1].astype(int) == _DRUG[dr])  # the drawn drug
        assert np.all(rows[:, 2] == 0.0)  # not control
        # condition embedding is the drawn perturbed leaf
        np.testing.assert_array_equal(out["condition"]["drug"], [[_LINE[cl], _DRUG[dr]]])

    assert set(seen_ctx) == set(LINES)  # each control population once


def test_n_conditions_cycles_control_populations():
    adata = _adata()
    loader = DAGEvalLoader(_scheme(adata), _CFG, _condition_fn)
    outs = list(loader.iter_conditions(n_conditions=5))
    assert len(outs) == 5  # 2 control populations cycled to 5 batches
    for out in outs:
        assert out["leaf"][1] != "control"


def test_deterministic_across_calls():
    adata = _adata()
    loader = DAGEvalLoader(_scheme(adata), _CFG, _condition_fn)
    a = [out["leaf"] for out in loader.iter_conditions(n_conditions=6)]
    b = [out["leaf"] for out in loader.iter_conditions(n_conditions=6)]
    assert a == b  # same seed ⇒ same drawn conditions each call


def test_reps_aligned_same_cells():
    adata = _adata()
    adata.obsm["rep"] = adata.X.copy()
    cols = ("cell_line", "drug")
    combos = {tuple(r) for r in adata.obs[list(cols)].to_numpy()}
    scheme = Scheme(
        sources={"data": adata},
        nodes={
            "pert": Node("data", cols, ("X", "obsm/rep"), uniform([c for c in combos if c[1] != "control"])),
            "ctrl": Node("data", cols, "X", uniform([c for c in combos if c[1] == "control"])),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=("cell_line",)),),
        seed=0,
    )
    out = next(DAGEvalLoader(scheme, _CFG, _condition_fn).iter_conditions())
    np.testing.assert_array_equal(np.asarray(out["target_reps"]["X"]), np.asarray(out["target_reps"]["obsm/rep"]))
