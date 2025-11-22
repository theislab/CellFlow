"""Data splitter for creating train/validation/test splits from TrainingData objects."""

import logging
from typing import Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from scaleflow.data._data import GroupedDistributionAnnotation

logger = logging.getLogger(__name__)


@dataclass
class AnnotationSplitter:
    annotation: GroupedDistributionAnnotation
    holdout_combinations: bool
    split_by: list[str]
    split_key: str
    force_training_values: dict[str, Any]
    ratios: list[float]
    random_state: int

    def __post_init__(self):
        if len(self.split_by) == 0:
            raise ValueError("split_by must be a non-empty list")
        if len(self.ratios) != 3:
            raise ValueError("ratios must be a list of 3 values [train, val, test]")
        if not np.isclose(sum(self.ratios), 1.0):
            raise ValueError("ratios must sum to 1.0")
        if not all(ratio > 0 and ratio < 1.0 for ratio in self.ratios):
            raise ValueError("ratios must be between 0.0 and 1.0")
        if not set(self.force_training_values.keys()).issubset(self.split_by):
            raise ValueError(
                f"force_training_values keys must be a subset of split_by: {self.force_training_values.keys()}"
            )
        self.train_ratio = self.ratios[0]
        self.val_ratio = self.ratios[1]
        self.test_ratio = self.ratios[2]

    def _calculate_split_sizes(self, total_combinations: int) -> tuple[int, int, int]:
        train_size = round(self.train_ratio * total_combinations)
        val_size = round(self.val_ratio * total_combinations)
        test_size_temp = round(self.test_ratio * total_combinations)
        if test_size_temp == 0 or val_size == 0 or train_size == 0:
            raise ValueError(
                "When multiplying the ratio by the total number of combinations, "
                "0 was encountered for one of the splits."
                "Either modify the ratios or the total number of combinations."
                f"The ratios are {self.train_ratio}, {self.val_ratio}, {self.test_ratio} and the total number of combinations is {total_combinations}."
                f"The calculated sizes are {train_size}, {val_size}, {test_size_temp}."
            )
        test_size = total_combinations - train_size - val_size
        return train_size, val_size, test_size

    @staticmethod
    def _check_df_unique_columns(df_unique: pd.DataFrame, combination_keys: list[str]) -> None:
        if not all(key in df_unique.columns for key in combination_keys):
            raise ValueError(f"combination_keys must be in df.columns: {combination_keys}")

    @staticmethod
    def _split_two(
        df_unique: pd.DataFrame,
        train_size: int,
        split_key: str,
        is_in_training_key: str | None = None,
        not_in_training_key: str | None = None,
        random_state: int = 42,
        overwrite: bool = False,
    ):
        if is_in_training_key is not None:
            if is_in_training_key not in df_unique.columns:
                raise ValueError(f"{is_in_training_key} must be in df.columns: {is_in_training_key}")
        if not_in_training_key is not None:
            if not_in_training_key not in df_unique.columns:
                raise ValueError(f"{not_in_training_key} must be in df.columns: {not_in_training_key}")
        if split_key in df_unique.columns and not overwrite:
            raise ValueError(f"{split_key} already in df.columns: {split_key} and overwrite is False")
        df_unique[split_key] = "test_val"

        # remove the forced training combinations from the unique combinations
        if is_in_training_key is not None:
            df_unique = df_unique[df_unique[is_in_training_key]]
        if not_in_training_key is not None:
            df_unique = df_unique[~df_unique[not_in_training_key]]
        if len(df_unique) == 0:
            raise ValueError(
                "There must be at least one unique combination of split_by keys after removing forced training combinations"
            )
        # shuffle the unique combinations
        df_unique = df_unique.sample(frac=1, random_state=random_state).reset_index(drop=True)

        df_unique.loc[: train_size - 1, split_key] = "train"

        return df_unique

    def _split_df(self):
        # calculate the sizes of the splits
        is_in_training_key = None
        df = self.annotation.src_tgt_dist_df.copy()
        df_unique = df.drop_duplicates(subset=self.split_by)
        AnnotationSplitter._check_df_unique_columns(df_unique, self.split_by)
        total_combinations = len(df_unique)
        train_size, val_size, test_size = self._calculate_split_sizes(
            total_combinations=total_combinations,
        )

        if self.holdout_combinations:
            is_in_training_key = "forced_in_train"
            AnnotationSplitter._contains_value(
                df_unique=df_unique,
                combination_keys=self.split_by,
                values=self.annotation.default_values,
                column_key=is_in_training_key,
                accept_nan=True,
            )
        if len(self.force_training_values) > 0:
            is_in_training_key = "forced_in_train"
            AnnotationSplitter._contains_value(
                df_unique=df_unique,
                combination_keys=self.split_by,
                values=self.force_training_values,
                column_key=is_in_training_key,
                accept_nan=False,
            )
        df_unique = self._split_two(
            df_unique=df_unique,
            train_size=train_size,
            random_state=self.random_state,
            is_in_training_key=is_in_training_key,
            not_in_training_key=None,
            split_key=self.split_key,
        )

        # Shuffle test_val and split into test and val
        test_val_mask = df_unique[self.split_key] == "test_val"
        test_val_df = df_unique[test_val_mask].copy()
        test_val_df = test_val_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Split test_val into test and val
        test_val_df.loc[: test_size - 1, self.split_key] = "test"
        test_val_df.loc[test_size:, self.split_key] = "val"

        # Update the original dataframe with the new split labels
        df_unique.loc[test_val_mask, self.split_key] = test_val_df[self.split_key].values
        return df_unique

    @staticmethod
    def _contains_value(
        df_unique: pd.DataFrame,
        combination_keys: list[str],
        values: dict[str, Any],
        column_key: str,
        accept_nan: bool,
    ):
        """
        Check if the values are in the dataframe, then write it to the column_key.

        If the column_key is already there then just use an OR operator to add the values.
        If column_key is not there then initialize with False

        Parameters
        ----------
        df_unique: pd.DataFrame
            The dataframe to check the values in.
        combination_keys: list[str]
            The keys to check the values in.
        values: dict[str, Any]
            The values to check the values in.
        column_key: str
            The column to write the values to.
        accept_nan: bool
            Whether to accept NaN values.
        """
        AnnotationSplitter._check_df_unique_columns(
            df_unique=df_unique,
            combination_keys=combination_keys,
        )

        # Initialize column_key if it doesn't exist
        if column_key not in df_unique.columns:
            df_unique[column_key] = False

        # For each key-value pair in values dict
        for key, value in values.items():
            if key not in combination_keys:
                continue

            # Check if the row matches the value or is NaN (if accept_nan is True)
            if accept_nan:
                mask = (df_unique[key] == value) | (pd.isna(df_unique[key]))
            else:
                mask = df_unique[key] == value

            # Use OR operator to update column_key
            df_unique[column_key] = df_unique[column_key] | mask
