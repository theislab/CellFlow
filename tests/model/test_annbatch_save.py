"""save/load and get_condition_embedding for annbatch-path models.

`save` pickles the whole model incl. the streaming loaders. The loaders drop their live annbatch
iterators on pickle (generators aren't picklable) but keep the RNG/schedule state, so a reloaded model
resumes the same reproducible stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad

import cellflow
from dagloader import SamplerConfig

_PREP = {
    "sample_rep": "X",
    "control_key": "control",
    "perturbation_covariates": {"drug": ["drug"]},
    "split_covariates": ["cell_line"],
}
_CFG = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8)


def _adata():
    rng = np.random.default_rng(0)
    rows = [(cl, dr) for cl in ("A", "B") for dr in ("control", "d1", "d2") for _ in range(16)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == "control"
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=rng.normal(size=(len(obs), 5)).astype("float32"), obs=obs)


def _prepared(seed=7):
    cf = cellflow.model.CellFlow()
    cf.prepare_annbatch_data(source=_adata(), sampler_config=_CFG, seed=seed, **_PREP)
    return cf


def _stream(cf, n):
    return [cf._dataloader.sample()["tgt_cell_data"].copy() for _ in range(n)]


class TestAnnbatchSave:
    def test_save_load_after_prepare(self, tmp_path):
        cf = _prepared()
        cf.save(str(tmp_path), file_prefix="p", overwrite=True)
        loaded = cellflow.model.CellFlow.load(str(tmp_path / "p_CellFlow.pkl"))
        assert loaded._scheme is not None
        assert loaded._dataloader.sample()["tgt_cell_data"].shape == (8, 5)

    def test_save_load_mid_stream_resumes_deterministically(self, tmp_path):
        # advance both models to the same point, save, load — the resumed streams must match
        a, b = _prepared(), _prepared()
        _stream(a, 2)
        _stream(b, 2)
        a.save(str(tmp_path), file_prefix="a", overwrite=True)
        b.save(str(tmp_path), file_prefix="b", overwrite=True)
        la = cellflow.model.CellFlow.load(str(tmp_path / "a_CellFlow.pkl"))
        lb = cellflow.model.CellFlow.load(str(tmp_path / "b_CellFlow.pkl"))
        ra, rb = _stream(la, 3), _stream(lb, 3)
        assert all(np.array_equal(x, y) for x, y in zip(ra, rb, strict=True))
        # and the resumed state is preserved (not reset to the seed → differs from a fresh model)
        fresh = _stream(_prepared(), 3)
        assert not all(np.array_equal(x, y) for x, y in zip(ra, fresh, strict=True))

    def test_save_load_after_training(self, tmp_path):
        cf = _prepared()
        cf.prepare_model(
            pooling="mean", condition_embedding_dim=8, time_encoder_dims=(8,), hidden_dims=(8,), decoder_dims=(8,)
        )
        cf.train(num_iterations=2, valid_freq=100)
        cf.save(str(tmp_path), file_prefix="t", overwrite=True)
        loaded = cellflow.model.CellFlow.load(str(tmp_path / "t_CellFlow.pkl"))
        assert loaded.solver is not None and loaded._dataloader is not None
        assert loaded._dataloader.sample()["tgt_cell_data"].shape == (8, 5)  # loader still usable

    def test_get_condition_embedding_without_adata_warns(self, tmp_path):
        cf = _prepared()
        cf.prepare_model(
            pooling="mean", condition_embedding_dim=8, time_encoder_dims=(8,), hidden_dims=(8,), decoder_dims=(8,)
        )
        cf.train(num_iterations=2, valid_freq=100)
        cov = _adata().obs.drop_duplicates(subset=["cell_line", "drug"])  # carries the control column too
        with pytest.warns(UserWarning, match="streaming path"):
            df_mean, df_var = cf.get_condition_embedding(cov)  # default key_added, no adata → warns, no crash
        assert isinstance(df_mean, pd.DataFrame) and isinstance(df_var, pd.DataFrame)
        assert len(df_mean) == len(cov)  # one embedding row per provided condition
