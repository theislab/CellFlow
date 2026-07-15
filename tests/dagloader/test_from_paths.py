"""``Scheme.from_paths`` — resolving zarr paths (single adata / collection root / list) to sources.

Covers the three source-value shapes the classmethod adds on top of the constructor, the auto-detection
of a single path (adata vs annbatch collection root), the "load only the reps the nodes use" contract,
and that a resolved scheme streams matched batches end-to-end just like a constructed one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import warnings

import anndata as ad
import scipy.sparse as sp
from annbatch import DatasetCollection

from dagloader import Bind, DAGLoader, Node, SamplerConfig, Scheme, uniform
from dagloader._io import load_backed_adata

LINES = ("A", "B")
DRUGS = ("control", "d1", "d2")
COLS = ("cell_line", "drug")


def _adata(n_per_combo: int = 16, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    rows = [(cl, dr) for cl in LINES for dr in DRUGS for _ in range(n_per_combo)]
    obs = pd.DataFrame(rows, columns=list(COLS))
    obs["extra"] = "unused"  # an obs column no node references — must not be required
    obs.index = obs.index.astype(str)
    a = ad.AnnData(X=sp.csr_matrix(rng.random((len(obs), 5), dtype="float32")), obs=obs)
    a.obsm["emb"] = rng.random((len(obs), 3), dtype="float32")
    a.layers["log1p"] = sp.csr_matrix(rng.random((len(obs), 5), dtype="float32"))
    return a


def _write_zarr(adata: ad.AnnData, path) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zarr v2/v3 default-format UserWarning
        adata.write_zarr(str(path))
    return str(path)


def _weights():
    combos = {(cl, dr) for cl in LINES for dr in DRUGS}
    pert = uniform([c for c in combos if c[1] != "control"])
    ctrl = uniform([c for c in combos if c[1] == "control"])
    return pert, ctrl


def _cfg() -> SamplerConfig:
    return SamplerConfig(batch_size=16, chunk_size=16, preload_nchunks=1, to=None)


# ── load_backed_adata: only the requested reps/cols are materialized ───────────────────────────


def test_load_backed_adata_reads_only_requested_keys(tmp_path):
    g = _open_group(_write_zarr(_adata(), tmp_path / "a.zarr"))
    backed = load_backed_adata(g, keys=("X", "obsm/emb"), cols=COLS)

    assert isinstance(backed.X, ad.abc.CSRDataset)  # sparse rep stays backed (not read into RAM)
    assert list(backed.obsm) == ["emb"]  # obsm/emb requested → present
    assert "log1p" not in backed.layers  # layers/log1p not requested → never touched
    assert list(backed.obs.columns) == list(COLS)  # obs reduced to the requested cols
    np.testing.assert_array_equal(backed.obsm["emb"].shape, (backed.n_obs, 3))


def test_load_backed_adata_layers_and_no_x(tmp_path):
    g = _open_group(_write_zarr(_adata(), tmp_path / "a.zarr"))
    backed = load_backed_adata(g, keys=("layers/log1p",), cols=("cell_line",))

    assert backed.X is None  # X not requested
    assert "log1p" in backed.layers  # requested layer present
    assert list(backed.obs.columns) == ["cell_line"]


def _open_group(path):
    import zarr

    return zarr.open_group(path, mode="r")


# ── open_source / from_paths: the three source-value shapes ────────────────────────────────────


def test_from_paths_single_adata_path_autodetects_adata(tmp_path):
    pert, ctrl = _weights()
    p = _write_zarr(_adata(), tmp_path / "a.zarr")
    nodes = {"pert": Node("data", COLS, "X", pert), "ctrl": Node("data", COLS, ("X", "obsm/emb"), ctrl)}
    s = Scheme.from_paths(sources={"data": p}, nodes=nodes, root="pert", seed=0, binds=(Bind("pert", "ctrl", ("cell_line",)),))

    src = s.sources["data"]
    assert isinstance(src, ad.AnnData)
    assert isinstance(src.X, ad.abc.CSRDataset)  # backed, not in-memory
    assert list(src.obsm) == ["emb"]  # union of the nodes' keys → only obsm/emb
    assert "log1p" not in src.layers


def test_from_paths_collection_root_autodetects_dataset_collection(tmp_path):
    pert, ctrl = _weights()
    ap = _write_zarr(_adata(), tmp_path / "a.zarr")
    cp = str(tmp_path / "coll.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DatasetCollection(cp, mode="a").add_adatas([ap], groupby=list(COLS), shuffle=False)

    nodes = {"pert": Node("data", COLS, "X", pert), "ctrl": Node("data", COLS, "X", ctrl)}
    s = Scheme.from_paths(sources={"data": cp}, nodes=nodes, root="pert", seed=0, binds=(Bind("pert", "ctrl", ("cell_line",)),))
    assert isinstance(s.sources["data"], DatasetCollection)


def test_from_paths_list_of_paths_gives_list_of_backed_adata(tmp_path):
    pert, ctrl = _weights()
    paths = [_write_zarr(_adata(seed=i), tmp_path / f"a{i}.zarr") for i in range(2)]
    nodes = {"pert": Node("data", COLS, "X", pert), "ctrl": Node("data", COLS, "X", ctrl)}
    s = Scheme.from_paths(sources={"data": paths}, nodes=nodes, root="pert", seed=0, binds=(Bind("pert", "ctrl", ("cell_line",)),))

    src = s.sources["data"]
    assert isinstance(src, list) and len(src) == 2
    assert all(isinstance(a, ad.AnnData) for a in src)


def test_from_paths_passes_through_constructed_containers(tmp_path):
    pert, ctrl = _weights()
    adata = _adata()  # already in-memory AnnData
    nodes = {"pert": Node("data", COLS, "X", pert), "ctrl": Node("data", COLS, "X", ctrl)}
    s = Scheme.from_paths(sources={"data": adata}, nodes=nodes, root="pert", seed=0, binds=(Bind("pert", "ctrl", ("cell_line",)),))
    assert s.sources["data"] is adata  # unchanged


# ── end-to-end: a from_paths scheme streams matched batches ────────────────────────────────────


@pytest.mark.parametrize(
    "make_source",
    [
        pytest.param(lambda tp: _write_zarr(_adata(), tp / "a.zarr"), id="single-path"),
        pytest.param(lambda tp: [_write_zarr(_adata(seed=i), tp / f"a{i}.zarr") for i in range(2)], id="list-of-paths"),
    ],
)
def test_from_paths_streams_matched_batches(tmp_path, make_source):
    pert, ctrl = _weights()
    src = make_source(tmp_path)
    nodes = {"pert": Node("data", COLS, "X", pert), "ctrl": Node("data", COLS, ("X", "obsm/emb"), ctrl)}
    s = Scheme.from_paths(sources={"data": src}, nodes=nodes, root="pert", seed=0, binds=(Bind("pert", "ctrl", ("cell_line",)),))

    batch = next(iter(DAGLoader(s, _cfg())))
    assert batch["target"].shape == (16, 5)
    assert batch["source"].shape == (16, 5)  # matched control rows
    assert batch["source_reps"]["obsm/emb"].shape == (16, 3)  # aligned obsm rep of the same cells


# ── Node.keys accepts anndata.acc accessors (normalized to loc strings) ────────────────────────


def test_node_keys_accept_anndata_accessors():
    from anndata.acc import A

    from dagloader import Node

    # single accessor and a tuple of accessors both normalize to the loc-string form
    assert Node("s", COLS, A.X).keys == ("X",)
    assert Node("s", COLS, A.obsm["emb"]).keys == ("obsm/emb",)
    assert Node("s", COLS, A.layers["log1p"]).keys == ("layers/log1p",)
    assert Node("s", COLS, (A.X, A.obsm["emb"])).keys == ("X", "obsm/emb")
    # mixed accessor + legacy string is fine
    assert Node("s", COLS, (A.X, "layers/log1p")).keys == ("X", "layers/log1p")


def test_from_paths_streams_with_accessor_keys(tmp_path):
    from anndata.acc import A

    pert, ctrl = _weights()
    p = _write_zarr(_adata(), tmp_path / "a.zarr")
    # describe the spots with accessors instead of "X" / "obsm/emb" strings
    nodes = {"pert": Node("data", COLS, A.X, pert), "ctrl": Node("data", COLS, (A.X, A.obsm["emb"]), ctrl)}
    s = Scheme.from_paths(sources={"data": p}, nodes=nodes, root="pert", seed=0, binds=(Bind("pert", "ctrl", ("cell_line",)),))

    assert list(s.sources["data"].obsm) == ["emb"]  # accessor-described rep resolved & loaded
    batch = next(iter(DAGLoader(s, _cfg())))
    assert batch["source_reps"]["obsm/emb"].shape == (16, 3)  # keyed by the normalized loc string
