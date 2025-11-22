from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

import anndata as ad
from scaleflow.data._utils import write_sharded, write_dist_data_threaded

import pandas as pd

__all__ = [
    "GroupedDistribution",
    "GroupedDistributionData",
    "GroupedDistributionAnnotation",
]


@dataclass
class ReturnData:  # TODO: this should rather be a NamedTuple
    split_covariates_mask: np.ndarray | None
    split_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_covariates_mask: np.ndarray | None
    perturbation_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, np.ndarray]
    control_to_perturbation: dict[int, np.ndarray]
    max_combination_length: int


class BaseDataMixin:
    """Base class for data containers."""

    @property
    def n_controls(self) -> int:
        """Returns the number of control covariate values."""
        return len(self.split_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbations(self) -> int:
        """Returns the number of perturbation covariate combinations."""
        return len(self.perturbation_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbation_covariates(self) -> int:
        """Returns the number of perturbation covariates."""
        return len(self.condition_data)  # type: ignore[attr-defined]

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "n_controls": self.n_controls,
            "n_perturbations": self.n_perturbations,
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"


@dataclass
class GroupedDistributionData:
    src_to_tgt_dist_map: dict[int, list[int]]  # (n_src_dists) → (n_tgt_dists_{src_dist_idx})
    src_data: dict[int, np.ndarray]  # (n_src_dists) → (n_cells_{src_dist_idx}, n_features)
    tgt_data: dict[int, np.ndarray]  # (n_tgt_dists) → (n_cells_{tgt_dist_idx}, n_features)
    conditions: dict[int, np.ndarray]  # (n_tgt_dists) → (n_cond_features_1, n_cond_features_2)

    @classmethod
    def read_zarr(
        cls,
        group: zarr.Group,
    ) -> GroupedDistributionData:
        """
        Read the grouped distribution data from a Zarr group.
        """
        return cls(
            src_to_tgt_dist_map={
                int(k): np.array(group["src_to_tgt_dist_map"][k]) for k in group["src_to_tgt_dist_map"].keys()
            },
            src_data={int(k): group["src_data"][k] for k in group["src_data"].keys()},
            tgt_data={int(k): group["tgt_data"][k] for k in group["tgt_data"].keys()},
            conditions={int(k): np.array(group["conditions"][k]) for k in group["conditions"].keys()},
        )

    def write_zarr_group(
        self,
        group: zarr.Group,
        chunk_size: int,
        shard_size: int,
        max_workers: int,
    ) -> None:
        """
        Write the grouped distribution data to a Zarr group.
        """
        data = group.create_group("data")
        write_sharded(
            group=data,
            name="src_to_tgt_dist_map",
            data={str(k): np.array(v) for k, v in self.src_to_tgt_dist_map.items()},
            chunk_size=chunk_size,
            shard_size=shard_size,
            compressors=None,
        )
        to_write = {
            "src_data": self.src_data,
            "tgt_data": self.tgt_data,
            "conditions": self.conditions,
        }
        for key, value in to_write.items():
            sub_group = data.create_group(key)
            write_dist_data_threaded(
                group=sub_group,
                dist_data=value,
                chunk_size=chunk_size,
                shard_size=shard_size,
                max_workers=max_workers,
            )
        return None


@dataclass
class GroupedDistributionAnnotation:
    old_obs_index: np.ndarray  # (n_cells,) to be able to map back to the original index

    src_dist_idx_to_labels: dict[int, Any]  # (n_src_dists) → Any (e.g. list of strings)
    tgt_dist_idx_to_labels: dict[int, Any]  # (n_tgt_dists) → Any (e.g. list of strings)
    src_tgt_dist_df: pd.DataFrame

    default_values: dict[str, Any]
    tgt_dist_keys: list[str]
    src_dist_keys: list[str]
    dist_flag_key: str

    @classmethod
    def read_zarr(
        cls,
        group: zarr.Group,
    ) -> GroupedDistributionAnnotation:
        """
        Read the grouped distribution annotation from a Zarr group.
        """
        elem = ad.io.read_elem(group)
        return cls(
            old_obs_index=elem["old_obs_index"],
            src_dist_idx_to_labels=elem["src_dist_idx_to_labels"],
            tgt_dist_idx_to_labels=elem["tgt_dist_idx_to_labels"],
            src_tgt_dist_df=elem["src_tgt_dist_df"],
            default_values=elem["default_values"],
            tgt_dist_keys=elem["tgt_dist_keys"],
            src_dist_keys=elem["src_dist_keys"],
            dist_flag_key=elem["dist_flag_key"],
        )

    def write_zarr_group(
        self,
        group: zarr.Group,
        chunk_size: int,
        shard_size: int,
    ) -> None:
        """
        Write the grouped distribution annotation to a Zarr group.
        """
        to_write = {
            "old_obs_index": self.old_obs_index,
            "src_dist_idx_to_labels": {str(k): np.array(v) for k, v in self.src_dist_idx_to_labels.items()},
            "tgt_dist_idx_to_labels": {str(k): np.array(v) for k, v in self.tgt_dist_idx_to_labels.items()},
            "src_tgt_dist_df": self.src_tgt_dist_df,
            "control_values": self.control_values,
        }
        write_sharded(
            group=group,
            name="annotation",
            data=to_write,
            chunk_size=chunk_size,
            shard_size=shard_size,
            compressors=None,
        )
        return None

    def filter_by_tgt_dist_indices(self, tgt_dist_indices: list[int]) -> GroupedDistributionAnnotation:
        """
        Create a new GroupedDistributionAnnotation containing only the specified target distribution indices.

        Parameters
        ----------
        tgt_dist_indices : list[int]
            List of target distribution indices to include

        Returns
        -------
        GroupedDistributionAnnotation
            New annotation with filtered data
        """
        tgt_dist_indices_set = set(tgt_dist_indices)

        # Filter dataframe
        filtered_df = self.src_tgt_dist_df[self.src_tgt_dist_df["tgt_dist_idx"].isin(tgt_dist_indices_set)].copy()

        # Get involved source distributions
        involved_src_dists = set(filtered_df["src_dist_idx"].unique())

        # Filter labels
        filtered_tgt_labels = {
            tgt_idx: self.tgt_dist_idx_to_labels[tgt_idx]
            for tgt_idx in tgt_dist_indices
            if tgt_idx in self.tgt_dist_idx_to_labels
        }

        filtered_src_labels = {
            src_idx: self.src_dist_idx_to_labels[src_idx]
            for src_idx in involved_src_dists
            if src_idx in self.src_dist_idx_to_labels
        }

        return GroupedDistributionAnnotation(
            old_obs_index=self.old_obs_index,  # Keep original mapping
            src_dist_idx_to_labels=filtered_src_labels,
            tgt_dist_idx_to_labels=filtered_tgt_labels,
            src_tgt_dist_df=filtered_df,
            control_values=self.control_values,
        )


@dataclass
class GroupedDistribution:
    data: GroupedDistributionData
    annotation: GroupedDistributionAnnotation

    def write_zarr(
        self,
        path: str,
        *,
        chunk_size: int = 4096,
        shard_size: int = 65536,
        max_workers: int = 8,
    ) -> None:
        """Write the grouped distribution to a Zarr group."""
        ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr

        zgroup = zarr.open_group(path, mode="w")

        self.data.write_zarr_group(
            group=zgroup,
            chunk_size=chunk_size,
            shard_size=shard_size,
            max_workers=max_workers,
        )
        self.annotation.write_zarr_group(
            group=zgroup,
            chunk_size=chunk_size,
            shard_size=shard_size,
        )
        return None

    @classmethod
    def read_zarr(
        cls,
        path: str,
    ) -> GroupedDistribution:
        """Read the grouped distribution from a Zarr group."""
        zgroup = zarr.open_group(path, mode="r")
        annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])
        data = GroupedDistributionData.read_zarr(zgroup["data"])
        return cls(
            annotation=annotation,
            data=data,
        )

    def split_by_dist_df(self, dist_df: pd.DataFrame, column: str) -> dict[str, GroupedDistributionData]:
        """Split the grouped distribution by the given distribution dataframe."""
        if column not in dist_df.columns:
            raise ValueError(f"Column {column} not found in dist_df.")
        # assert categorical,boolean, or string
        if (
            not pd.api.types.is_categorical_dtype(dist_df[column])
            and not pd.api.types.is_bool_dtype(dist_df[column])
            and not pd.api.types.is_string_dtype(dist_df[column])
        ):
            raise ValueError(f"Column {column} must be categorical, boolean, or string.")

        split_values = dist_df[column].unique()
        # get the src_dist_idx and tgt_dist_idx for each value
        split_data = {}
        for value in split_values:
            filtered_df = dist_df.loc[dist_df[column] == value]
            # group by to map src_dist_idx and tgt_dist_idx
            src_tgt_dist_map = (
                filtered_df[["src_dist_idx", "tgt_dist_idx"]]
                .groupby("src_dist_idx")["tgt_dist_idx"]
                .apply(list)
                .to_dict()
            )
            src_data = {int(k): self.data.src_data[k] for k in src_tgt_dist_map.keys()}
            tgt_data = {int(k): self.data.tgt_data[k] for k in src_tgt_dist_map.keys()}
            conditions = {int(k): self.data.conditions[k] for k in src_tgt_dist_map.keys()}
            split_data[value] = GroupedDistributionData(
                src_to_tgt_dist_map=src_tgt_dist_map,
                src_data=src_data,
                tgt_data=tgt_data,
                conditions=conditions,
            )
        return split_data

    def filter_by_tgt_dist_indices(self, tgt_dist_indices: list[int]) -> GroupedDistribution:
        """
        Create a new GroupedDistribution containing only the specified target distribution indices.

        Parameters
        ----------
        tgt_dist_indices : list[int]
            List of target distribution indices to include

        Returns
        -------
        GroupedDistribution
            New GroupedDistribution with filtered data
        """
        tgt_dist_indices_set = set(tgt_dist_indices)

        # Filter annotation data
        filtered_df = self.annotation.src_tgt_dist_df[
            self.annotation.src_tgt_dist_df["tgt_dist_idx"].isin(tgt_dist_indices_set)
        ].copy()

        # Get involved source distributions
        involved_src_dists = set(filtered_df["src_dist_idx"].unique())

        # Filter data structures
        filtered_src_to_tgt = {
            src_idx: [tgt_idx for tgt_idx in tgt_list if tgt_idx in tgt_dist_indices_set]
            for src_idx, tgt_list in self.data.src_to_tgt_dist_map.items()
            if src_idx in involved_src_dists
        }
        # Remove empty mappings
        filtered_src_to_tgt = {k: v for k, v in filtered_src_to_tgt.items() if len(v) > 0}

        filtered_src_data = {src_idx: self.data.src_data[src_idx] for src_idx in filtered_src_to_tgt.keys()}

        filtered_tgt_data = {
            tgt_idx: self.data.tgt_data[tgt_idx] for tgt_idx in tgt_dist_indices if tgt_idx in self.data.tgt_data
        }

        filtered_conditions = {
            tgt_idx: self.data.conditions[tgt_idx] for tgt_idx in tgt_dist_indices if tgt_idx in self.data.conditions
        }

        filtered_tgt_labels = {
            tgt_idx: self.annotation.tgt_dist_idx_to_labels[tgt_idx]
            for tgt_idx in tgt_dist_indices
            if tgt_idx in self.annotation.tgt_dist_idx_to_labels
        }

        filtered_src_labels = {
            src_idx: self.annotation.src_dist_idx_to_labels[src_idx]
            for src_idx in filtered_src_to_tgt.keys()
            if src_idx in self.annotation.src_dist_idx_to_labels
        }

        # Note: old_obs_index remains the same as it maps to original data

        return GroupedDistribution(
            data=GroupedDistributionData(
                src_to_tgt_dist_map=filtered_src_to_tgt,
                src_data=filtered_src_data,
                tgt_data=filtered_tgt_data,
                conditions=filtered_conditions,
            ),
            annotation=GroupedDistributionAnnotation(
                old_obs_index=self.annotation.old_obs_index,
                src_dist_idx_to_labels=filtered_src_labels,
                tgt_dist_idx_to_labels=filtered_tgt_labels,
                src_tgt_dist_df=filtered_df,
            ),
        )
