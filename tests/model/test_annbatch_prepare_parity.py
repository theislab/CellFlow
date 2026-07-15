"""Parity guard for the optimized ``build_annbatch_training`` (streaming-path prepare).

The prepare step now deduplicates the grouping columns ONCE and drives the whole covariate encoder off
that tiny frame (instead of the full ~10^8-row obs). This test pins the outputs to the canonical
in-memory :class:`~cellflow.data._datamanager.DataManager` path — the ground truth the streaming path
has always had to match — plus a naive full-obs computation of the scheme's leaves. Any divergence in
``condition_data`` / scheme leaves / ``data_dim`` / ``condition_fn`` wiring is a regression.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

from cellflow.data._annbatch import build_annbatch_training
from cellflow.data._condition import _key_layout
from cellflow.data._datamanager import DataManager

# (perturbation_covariates, perturbation_covariate_reps, split_covariates, sample_covariates, sample_reps)
SPECS = [
    ({"drug": ["drug1"]}, {"drug": "drug"}, ["cell_type"], [], {}),  # embedding primary + split
    ({"drug": ["drug1"]}, {}, ["cell_type"], [], {}),  # one-hot primary + split
    ({"drug": ["drug1", "drug2"]}, {"drug": "drug"}, ["cell_type"], [], {}),  # combination length 2
    ({"drug": ["drug1"]}, {"drug": "drug"}, [], ["cell_type"], {"cell_type": "cell_type"}),  # sample-covar path
    ({"drug": ["drug1"], "dosage": ["dosage_a"]}, {"drug": "drug"}, ["cell_type"], [], {}),  # numeric linked covar
    ({"drug": ["drug1"]}, {}, [], [], {}),  # bare one-hot, no split/sample
]


def _grouping_cols(pert_covars, split_covars, samp_covars):
    """The build's deduped/ordered grouping columns (mirrors ``build_annbatch_training``)."""
    pert_cols = tuple(c for grp in pert_covars.values() for c in grp)
    return tuple(dict.fromkeys((*tuple(split_covars), *pert_cols, *tuple(samp_covars))))


def _build(adata, pert_covars, pert_reps, split_covars, samp_covars, samp_reps):
    return build_annbatch_training(
        data=adata,
        sample_rep="X",
        control_key="control",
        perturbation_covariates=pert_covars,
        perturbation_covariate_reps=pert_reps or None,
        split_covariates=split_covars,
        sample_covariates=samp_covars,
        sample_covariate_reps=samp_reps or None,
        rep_dict=adata.uns,
    )


def _reference_dm(adata, pert_covars, pert_reps, split_covars, samp_covars, samp_reps):
    dm = DataManager(
        adata,
        sample_rep="X",
        control_key="control",
        perturbation_covariates=pert_covars,
        perturbation_covariate_reps=pert_reps or None,
        split_covariates=split_covars,
        sample_covariates=samp_covars,
        sample_covariate_reps=samp_reps or None,
    )
    return dm, dm._get_condition_data(adata=adata)


@pytest.mark.parametrize(("pert_covars", "pert_reps", "split_covars", "samp_covars", "samp_reps"), SPECS)
class TestBuildAnnbatchParity:
    def test_condition_data_matches_in_memory(
        self, adata_perturbation: ad.AnnData, pert_covars, pert_reps, split_covars, samp_covars, samp_reps
    ):
        built = _build(adata_perturbation, pert_covars, pert_reps, split_covars, samp_covars, samp_reps)
        dm, ref = _reference_dm(adata_perturbation, pert_covars, pert_reps, split_covars, samp_covars, samp_reps)

        assert set(built.condition_data) == set(ref.condition_data)
        for key in ref.condition_data:
            np.testing.assert_array_equal(built.condition_data[key], ref.condition_data[key])
        assert built.max_combination_length == dm.max_combination_length
        assert built.data_dim == adata_perturbation.n_vars

    def test_scheme_leaves_match_naive_full_obs(
        self, adata_perturbation: ad.AnnData, pert_covars, pert_reps, split_covars, samp_covars, samp_reps
    ):
        built = _build(adata_perturbation, pert_covars, pert_reps, split_covars, samp_covars, samp_reps)
        cols = _grouping_cols(pert_covars, split_covars, samp_covars)
        obs = adata_perturbation.obs
        ctrl = obs["control"].to_numpy().astype(bool)

        def _leaves(mask):  # naive: dedup the FULL obs (the pre-optimization computation)
            return {tuple(map(str, r)) for r in obs.loc[mask, list(cols)].drop_duplicates().to_numpy()}

        pert_got = {tuple(map(str, k)) for k in built.scheme.nodes["pert"].weights}
        ctrl_got = {tuple(map(str, k)) for k in built.scheme.nodes["ctrl"].weights}
        assert pert_got == _leaves(~ctrl)
        assert ctrl_got == _leaves(ctrl)

    def test_condition_fn_maps_each_leaf_to_its_condition(
        self, adata_perturbation: ad.AnnData, pert_covars, pert_reps, split_covars, samp_covars, samp_reps
    ):
        built = _build(adata_perturbation, pert_covars, pert_reps, split_covars, samp_covars, samp_reps)
        dm, ref = _reference_dm(adata_perturbation, pert_covars, pert_reps, split_covars, samp_covars, samp_reps)
        cols = _grouping_cols(pert_covars, split_covars, samp_covars)
        # canonical tuple layout + reprojection cols->tuple_keys (same as DataManager / build)
        _, tuple_keys = _key_layout(dm._perturb_covar_keys, list(split_covars), list(samp_covars))
        reorder = [cols.index(c) for c in tuple_keys]
        cov_to_idx = {tuple(map(str, v)): k for k, v in ref.perturbation_idx_to_covariates.items()}

        for leaf in built.scheme.nodes["pert"].weights:  # every perturbed leaf
            idx = cov_to_idx[tuple(str(leaf[i]) for i in reorder)]
            emitted = built.condition_fn(leaf)
            assert set(emitted) == set(ref.condition_data)
            for group in ref.condition_data:
                np.testing.assert_array_equal(emitted[group], ref.condition_data[group][[idx]])


def _tahoe_shaped_adata(n_lines=5, n_drugs=6, per_combo=8, seed=0) -> ad.AnnData:
    """Tahoe-like obs: cell_line[category] / drug[object-str] / is_control[bool] (exercises the cast)."""
    rng = np.random.default_rng(seed)
    lines = [f"CL{i}" for i in range(n_lines)]
    drugs = ["control"] + [f"drug{i}" for i in range(n_drugs - 1)]
    rows = [(cl, dr) for cl in lines for dr in drugs for _ in range(per_combo)]
    rows = [rows[i] for i in rng.permutation(len(rows))]  # shuffled → build must sort in-memory
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["cell_line"] = obs["cell_line"].astype("category")
    obs["drug"] = obs["drug"].astype(object)  # object/str — NOT pre-categorical
    obs["control"] = (obs["drug"] == "control").to_numpy()
    obs.index = obs.index.astype(str)
    X = rng.normal(size=(len(obs), 4)).astype("float32")
    return ad.AnnData(X=X, obs=obs)


@pytest.mark.parametrize("pert_reps", [{"drug": "drug_emb"}, {}])
def test_tahoe_shaped_object_drug_parity(pert_reps):
    adata = _tahoe_shaped_adata()
    if pert_reps:
        drugs = list(pd.unique(adata.obs["drug"]))
        adata.uns["drug_emb"] = {d: np.random.default_rng(1).normal(size=6).astype("float32") for d in drugs}
    built = build_annbatch_training(
        data=adata,
        sample_rep="X",
        control_key="control",
        perturbation_covariates={"drug": ["drug"]},
        perturbation_covariate_reps=pert_reps or None,
        split_covariates=["cell_line"],
        rep_dict=adata.uns,
    )
    dm = DataManager(
        adata,
        sample_rep="X",
        control_key="control",
        perturbation_covariates={"drug": ["drug"]},
        perturbation_covariate_reps=pert_reps or None,
        split_covariates=["cell_line"],
    )
    ref = dm._get_condition_data(adata=adata)
    assert set(built.condition_data) == set(ref.condition_data)
    for key in ref.condition_data:
        np.testing.assert_array_equal(built.condition_data[key], ref.condition_data[key])
    # scheme leaves == naive full-obs dedup
    obs = adata.obs
    ctrl = obs["control"].to_numpy().astype(bool)
    pert_got = {tuple(map(str, k)) for k in built.scheme.nodes["pert"].weights}
    pert_ref = {tuple(map(str, r)) for r in obs.loc[~ctrl, ["cell_line", "drug"]].drop_duplicates().to_numpy()}
    assert pert_got == pert_ref
