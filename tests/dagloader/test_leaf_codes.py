"""`leaf_codes` is vectorized (factorize + category cast) — it must stay byte-identical to the
original per-cell Python-loop implementation across dtypes and edge cases. This guards that rewrite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dagloader._io import leaf_codes


def _ref_leaf_codes(obs: pd.DataFrame, cols):
    """The original O(n_cells) implementation, kept here as the parity oracle."""
    tuples = [tuple(row) for row in obs[list(cols)].to_numpy()]
    leaves = sorted(set(tuples), key=lambda t: tuple(map(str, t)))
    code_of = {lf: i for i, lf in enumerate(leaves)}
    return np.array([code_of[t] for t in tuples], dtype=np.int64), leaves


def _obs(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    lines = rng.choice([f"CL{i}" for i in range(7)], n)
    drugs = rng.choice(["control", "d1", "d2", "d3", "d4"], n)
    df = pd.DataFrame(
        {
            "cell_line_cat": pd.Categorical(lines),  # categorical
            "drug_obj": drugs.astype(object),  # object/str
            "control_bool": (drugs == "control"),  # bool
            "dose_int": rng.integers(0, 4, n),  # int
            "dose_float": rng.choice([0.5, 1.0, 2.0], n),  # float
        }
    )
    return df


COL_SETS = [
    ["cell_line_cat", "drug_obj"],
    ["drug_obj"],  # single object col
    ["cell_line_cat"],  # single categorical col
    ["control_bool"],  # single bool col
    ["dose_int", "dose_float"],  # numeric combo
    ["cell_line_cat", "drug_obj", "control_bool", "dose_int", "dose_float"],  # mixed, 5 cols
]


@pytest.mark.parametrize("cols", COL_SETS)
def test_matches_reference(cols):
    obs = _obs()
    codes, leaves = leaf_codes(obs, cols)
    rcodes, rleaves = _ref_leaf_codes(obs, cols)
    np.testing.assert_array_equal(codes, rcodes)
    assert leaves == rleaves  # exact tuples (same .to_numpy() element types), same order
    # codes index into leaves and round-trip to the cell's own combination
    got = [leaves[c] for c in codes]
    assert got == [tuple(r) for r in obs[cols].to_numpy()]


def test_matches_reference_with_nan():
    obs = _obs()
    obs.loc[obs.index[:20], "drug_obj"] = np.nan  # NaN treated as a real category, not a sentinel
    codes, leaves = leaf_codes(obs, ["cell_line_cat", "drug_obj"])
    rcodes, rleaves = _ref_leaf_codes(obs, ["cell_line_cat", "drug_obj"])
    np.testing.assert_array_equal(codes, rcodes)
    assert [tuple(map(str, t)) for t in leaves] == [tuple(map(str, t)) for t in rleaves]


def test_empty_obs():
    obs = _obs().iloc[:0]
    codes, leaves = leaf_codes(obs, ["cell_line_cat", "drug_obj"])
    assert codes.shape == (0,) and codes.dtype == np.int64
    assert leaves == []


def test_precast_categorical_matches_object():
    # casting a col to categorical up-front (as the Tahoe zarrs will) must not change the result
    obs = _obs()
    obs_cat = obs.assign(drug_obj=obs["drug_obj"].astype("category"))
    c1, l1 = leaf_codes(obs, ["cell_line_cat", "drug_obj"])
    c2, l2 = leaf_codes(obs_cat, ["cell_line_cat", "drug_obj"])
    np.testing.assert_array_equal(c1, c2)
    assert [tuple(map(str, t)) for t in l1] == [tuple(map(str, t)) for t in l2]
