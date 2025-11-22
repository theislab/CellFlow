from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pandas as pd

from scaleflow.data import (
    GroupedDistribution,
    GroupedDistributionAnnotation,
    GroupedDistributionData,
)
from scaleflow.logging import timer

from ._anndata_location import AnnDataLocation

__all__ = ["DataManager"]

if TYPE_CHECKING:
    import anndata


@dataclass
class DataManager:
    dist_flag_key: str
    src_dist_keys: list[str]
    tgt_dist_keys: list[str]
    data_location: AnnDataLocation
    rep_keys: dict[str, str] | None = None

    def __post_init__(
        self,
    ):
        self._verify_dist_keys(self.src_dist_keys)
        self._verify_dist_keys(self.tgt_dist_keys)
        # they shouldn't overlap
        if len(set(self.src_dist_keys) & set(self.tgt_dist_keys)) > 0:
            raise ValueError("Source and target distributions must not overlap.")
        if self.rep_keys is not None:
            if not set(self.rep_keys.keys()).issubset(set(self.src_dist_keys) | set(self.tgt_dist_keys)):
                raise ValueError(
                    "Representation locations must be a subset of the source and target distribution keys."
                )

    def prepare_data(
        self,
        adata: "anndata.AnnData",
        verbose: bool = False,
    ) -> GroupedDistribution:
        """
        Distribution flag key must be a boolean column.

        The src and tgt distribution keys are recommended to be categorical columns otherwise sorting will be slow.
        Resets the index of obs. saves new to old index
        """
        DataManager._verify_rep_keys_exists(self.rep_keys, adata)

        src_tgt_dist_keys = [*self.src_dist_keys, *self.tgt_dist_keys]

        cols = [self.dist_flag_key, *src_tgt_dist_keys]
        obs = adata.obs[cols].copy()
        old_index_mapping = obs.index.to_numpy()
        obs.reset_index(drop=True, inplace=True)

        # dtype must be boolean
        assert pd.api.types.is_bool_dtype(obs[self.dist_flag_key]), "Distribution flag key must be a boolean column."

        with timer("Sorting values", verbose=verbose):
            obs.sort_values(cols, inplace=True)

        obs["src_dist_idx"] = obs.groupby(self.src_dist_keys, observed=False).ngroup()
        dist_mask = ~obs[self.dist_flag_key]
        obs.loc[dist_mask, "tgt_dist_idx"] = obs.loc[dist_mask].groupby(src_tgt_dist_keys, observed=False).ngroup()

        # Fill NaN indices with a specific value before casting
        obs["src_dist_idx"] = obs["src_dist_idx"].fillna(-1).astype(np.int32)
        obs["tgt_dist_idx"] = obs["tgt_dist_idx"].fillna(-1).astype(np.int32)

        # prepare src_tgt_dist_df
        src_tgt_dist_df = obs.loc[~obs[self.dist_flag_key]]
        src_tgt_dist_df = src_tgt_dist_df[["src_dist_idx", "tgt_dist_idx", *src_tgt_dist_keys]]
        src_tgt_dist_df.drop_duplicates(inplace=True)

        # prepare default_values
        temp_df = obs.loc[obs[self.dist_flag_key]][self.tgt_dist_keys].drop_duplicates()
        if len(temp_df) != 1:
            raise ValueError("There should be exactly one control value.")
        default_values = temp_df.iloc[0].to_dict()

        # prepare src_to_tgt_dist_map
        src_to_tgt_dist_map = (
            src_tgt_dist_df[["src_dist_idx", "tgt_dist_idx"]]
            .groupby("src_dist_idx")["tgt_dist_idx"]
            .apply(list)
            .to_dict()
        )

        # prepare src_dist_labels
        src_dist_labels = (
            obs.loc[obs[self.dist_flag_key]][[*self.src_dist_keys, "src_dist_idx"]]
            .drop_duplicates()
            .set_index("src_dist_idx")
        )
        src_dist_labels = dict(
            zip(src_dist_labels.index, src_dist_labels.itertuples(index=False, name=None), strict=True)
        )

        # prepare tgt_dist_labels
        tgt_dist_labels = (
            obs.loc[~obs[self.dist_flag_key]][[*self.tgt_dist_keys, "tgt_dist_idx"]]
            .drop_duplicates()
            .set_index("tgt_dist_idx")
        )
        tgt_dist_labels = dict(zip(tgt_dist_labels.index, tgt_dist_labels.itertuples(index=False, name=None), strict=True))


        # prepare conditions and structure metadata
        col_to_repr = {key: adata.uns[self.rep_keys[key]] for key in self.rep_keys.keys()}

        # Compute condition_structure from first available label
        condition_structure = {}
        offset = 0
        first_src_label = next(iter(src_dist_labels.values()))
        first_tgt_label = next(iter(tgt_dist_labels.values()))

        for col, label in zip(self.src_dist_keys, first_src_label, strict=True):
            if col in col_to_repr:
                dim = len(col_to_repr[col][label])
                condition_structure[col] = (offset, offset + dim)
                offset += dim

        for col, label in zip(self.tgt_dist_keys, first_tgt_label, strict=True):
            if col in col_to_repr:
                dim = len(col_to_repr[col][label])
                condition_structure[col] = (offset, offset + dim)
                offset += dim
            elif isinstance(label, (int, float)):
                # Scalar value (like dosage)
                condition_structure[col] = (offset, offset + 1)
                offset += 1

        with timer("Getting conditions", verbose=verbose):
            conditions = {}
            for src_dist_idx, tgt_dist_idxs in src_to_tgt_dist_map.items():
                src_label = src_dist_labels[src_dist_idx]
                src_repr = [
                    DataManager._col_to_repr(col_to_repr, col, label)
                    for col, label in zip(self.src_dist_keys, src_label, strict=True)
                ]
                for tgt_dist_idx in tgt_dist_idxs:
                    tgt_label = tgt_dist_labels[tgt_dist_idx]
                    tgt_repr = [
                        DataManager._col_to_repr(col_to_repr, col, label)
                        for col, label in zip(self.tgt_dist_keys, tgt_label, strict=True)
                    ]
                    conditions[tgt_dist_idx] = np.concatenate([*src_repr, *tgt_repr])


        # prepare src_data and tgt_data
        arr = self.data_location(adata)
        if isinstance(arr, da.Array):
            arr = arr.compute()
        with timer("Getting source and target distribution data", verbose=verbose):
            src_dist_map = obs[obs[self.dist_flag_key]].groupby("src_dist_idx", observed=False).groups
            tgt_dist_map = obs[~obs[self.dist_flag_key]].groupby("tgt_dist_idx", observed=False).groups

            tgt_data = {int(k): arr[v.to_numpy()] for k, v in tgt_dist_map.items()}
            src_data = {int(k): arr[v.to_numpy()] for k, v in src_dist_map.items()}

        return GroupedDistribution(
            data=GroupedDistributionData(
                src_to_tgt_dist_map=src_to_tgt_dist_map,
                src_data=src_data,
                tgt_data=tgt_data,
                conditions=conditions,
            ),
            annotation=GroupedDistributionAnnotation(
                src_tgt_dist_df=src_tgt_dist_df,
                old_obs_index=old_index_mapping,
                tgt_dist_keys=self.tgt_dist_keys,
                src_dist_keys=self.src_dist_keys,
                dist_flag_key=self.dist_flag_key,
                src_dist_idx_to_labels=src_dist_labels,
                tgt_dist_idx_to_labels=tgt_dist_labels,
                default_values=default_values,
                condition_structure=condition_structure,
            ),
        )

    @staticmethod
    def _verify_dist_keys(dist_keys: list[str]) -> None:
        if len(dist_keys) == 0:
            raise ValueError("Number of distributions must be greater than 0.")
        # no duplicates
        if len(set(dist_keys)) != len(dist_keys):
            raise ValueError("Distributions must be unique.")

    @staticmethod
    def _verify_rep_keys_exists(rep_keys: dict[str, str], adata: "anndata.AnnData") -> None:
        for _, value in rep_keys.items():
            if value not in adata.uns:
                raise ValueError(f"Representation key {value} not found in adata.uns.")

    @staticmethod
    def _col_to_repr(col_to_repr: dict[str, dict[str, np.ndarray]], col: str, label: Any) -> np.ndarray:
        if col not in col_to_repr:
            # for example in case of dosage, we have a float label
            if isinstance(label, float):
                return np.array([label])
            raise ValueError(f"Column {col} not found in col_to_repr.")
        if label not in col_to_repr[col]:
            raise ValueError(f"Label {label} not found in col_to_repr[{col}].")
        return col_to_repr[col][label]
