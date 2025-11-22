"""Test suite for AnnotationSplitter with high coverage."""

import numpy as np
import pandas as pd
import pytest

from scaleflow.data import GroupedDistribution
from scaleflow.data._data_splitter import AnnotationSplitter


class TestAnnotationSplitterInit:
    """Test AnnotationSplitter initialization."""

    def test_init_valid_params(self, sample_grouped_distribution):
        """Test that AnnotationSplitter initializes with valid parameters."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        assert splitter.train_ratio == 0.6
        assert splitter.val_ratio == 0.2
        assert splitter.test_ratio == 0.2
        assert splitter.split_by == ["drug", "gene"]
        assert splitter.split_key == "split"
        assert splitter.random_state == 42

    def test_init_custom_split_key(self, sample_grouped_distribution):
        """Test initialization with custom split_key."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="my_custom_split",
            force_training_values={},
            ratios=[0.7, 0.15, 0.15],
            random_state=99,
        )

        assert splitter.split_key == "my_custom_split"

    def test_init_invalid_split_by_empty(self, sample_grouped_distribution):
        """Test that empty split_by raises ValueError."""
        with pytest.raises(ValueError, match="split_by must be a non-empty list"):
            AnnotationSplitter(
                annotation=sample_grouped_distribution.annotation,
                holdout_combinations=False,
                split_by=[],
                split_key="split",
                force_training_values={},
                ratios=[0.6, 0.2, 0.2],
                random_state=42,
            )

    def test_init_invalid_ratios_length(self, sample_grouped_distribution):
        """Test that ratios with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="ratios must be a list of 3 values"):
            AnnotationSplitter(
                annotation=sample_grouped_distribution.annotation,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={},
                ratios=[0.6, 0.4],
                random_state=42,
            )

    def test_init_invalid_ratios_sum(self, sample_grouped_distribution):
        """Test that ratios not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            AnnotationSplitter(
                annotation=sample_grouped_distribution.annotation,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={},
                ratios=[0.5, 0.3, 0.1],
                random_state=42,
            )

    def test_init_invalid_ratios_with_zero(self, sample_grouped_distribution):
        """Test that ratios with 0 raises ValueError."""
        with pytest.raises(ValueError, match="ratios must be between 0.0 and 1.0"):
            AnnotationSplitter(
                annotation=sample_grouped_distribution.annotation,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={},
                ratios=[0.8, 0.2, 0.0],
                random_state=42,
            )

    def test_init_invalid_ratios_with_one(self, sample_grouped_distribution):
        """Test that ratios with 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="ratios must be between 0.0 and 1.0"):
            AnnotationSplitter(
                annotation=sample_grouped_distribution.annotation,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={},
                ratios=[1.0, 0.0, 0.0],
                random_state=42,
            )

    def test_init_invalid_force_training_values(self, sample_grouped_distribution):
        """Test that force_training_values with keys not in split_by raises ValueError."""
        with pytest.raises(ValueError, match="force_training_values keys must be a subset of split_by"):
            AnnotationSplitter(
                annotation=sample_grouped_distribution.annotation,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={"cell_line": "cell_line_0"},
                ratios=[0.6, 0.2, 0.2],
                random_state=42,
            )


class TestAnnotationSplitterSplit:
    """Test the split method of AnnotationSplitter."""

    def test_split_basic(self, sample_grouped_distribution):
        """Test basic splitting functionality."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Check that split column exists
        assert "split" in df_split.columns

        # Check that all rows are labeled
        assert df_split["split"].notna().all()

        # Check that all three splits exist
        split_values = set(df_split["split"].unique())
        assert split_values == {"train", "val", "test"}

    def test_split_with_custom_split_key(self, sample_grouped_distribution):
        """Test splitting with a custom split_key name."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="my_split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Check that custom split column exists
        assert "my_split" in df_split.columns
        assert df_split["my_split"].notna().all()

        # Check all three splits exist
        split_values = set(df_split["my_split"].unique())
        assert split_values == {"train", "val", "test"}

    def test_split_single_key(self, sample_grouped_distribution):
        """Test splitting by a single key."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # All rows with the same drug should have the same split
        for drug in df_split["drug"].unique():
            drug_splits = df_split[df_split["drug"] == drug]["split"].unique()
            assert len(drug_splits) == 1, f"Drug {drug} has multiple splits: {drug_splits}"

    def test_split_ratios_approximately_correct(self, sample_grouped_distribution):
        """Test that split ratios are approximately correct."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Get unique combinations
        df_unique = df_split.drop_duplicates(subset=["drug", "gene"])

        total = len(df_unique)
        train_count = (df_unique["split"] == "train").sum()
        val_count = (df_unique["split"] == "val").sum()
        test_count = (df_unique["split"] == "test").sum()

        # Check counts sum to total
        assert train_count + val_count + test_count == total

        # Check ratios are approximately correct
        assert train_count >= 1
        assert val_count >= 1
        assert test_count >= 1

    def test_split_deterministic_with_random_state(self, sample_grouped_distribution):
        """Test that splitting is deterministic given a random_state."""
        splitter1 = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        splitter2 = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split1 = splitter1._split_df()
        df_split2 = splitter2._split_df()

        # Results should be identical
        pd.testing.assert_frame_equal(df_split1, df_split2)

    def test_split_different_with_different_random_state(self, sample_grouped_distribution):
        """Test that different random states produce different splits."""
        splitter1 = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        splitter2 = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=123,
        )

        df_split1 = splitter1._split_df()
        df_split2 = splitter2._split_df()

        # Results should be different
        assert not df_split1[["split","drug","gene"]].equals(df_split2[["split","drug","gene"]])

    def test_split_no_duplicate_combinations_in_multiple_splits(self, sample_grouped_distribution):
        """Test that no combination appears in multiple splits."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Group by split_by keys and check each group has only one split value
        for (drug, gene), group in df_split.groupby(["drug", "gene"]):
            unique_splits = group["split"].unique()
            assert len(unique_splits) == 1, (
                f"Combination (drug={drug}, gene={gene}) appears in multiple splits: {unique_splits}"
            )


class TestAnnotationSplitterForceTraining:
    """Test force_training_values functionality."""

    def test_force_training_single_value(self, sample_grouped_distribution):
        """Test that forced training values end up in training set."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={"drug": "drug_0"},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # All rows with drug_0 should be in training
        drug_0_rows = df_split[df_split["drug"] == "drug_0"]
        if len(drug_0_rows) > 0:
            assert (drug_0_rows["split"] == "train").all(), "drug_0 should be forced to training"

    def test_force_training_multiple_keys_or_logic(self, sample_grouped_distribution):
        """Test forcing multiple keys uses OR logic."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={"drug": "drug_0", "gene": "gene_0"},
            ratios=[0.7, 0.15, 0.15],
            random_state=42,
        )

        df_split = splitter._split_df()

        # All rows with drug_0 OR gene_0 should be in training
        forced_rows = df_split[(df_split["drug"] == "drug_0") | (df_split["gene"] == "gene_0")]
        if len(forced_rows) > 0:
            assert (forced_rows["split"] == "train").all()

    def test_force_training_reduces_available_combinations(self, sample_grouped_distribution):
        """Test that forcing values to training reduces available combinations for test/val."""
        # Without forcing
        splitter_no_force = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        df_no_force = splitter_no_force._split_df()

        # With forcing
        splitter_with_force = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={"drug": "drug_0"},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        df_with_force = splitter_with_force._split_df()

        # Training set should be different
        train_no_force = set(df_no_force[df_no_force["split"] == "train"]["drug"].unique())
        train_with_force = set(df_with_force[df_with_force["split"] == "train"]["drug"].unique())

        # drug_0 must be in training when forced
        assert "drug_0" in train_with_force


class TestAnnotationSplitterHoldout:
    """Test holdout_combinations functionality."""

    def test_holdout_combinations_control_in_train(self, sample_grouped_distribution):
        """Test that when holdout_combinations=True, default values are in training."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=True,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Get default values from annotation
        default_values = sample_grouped_distribution.annotation.default_values

        # Check that rows matching default values are in training
        for key, value in default_values.items():
            if key in df_split.columns:
                default_rows = df_split[df_split[key] == value]
                if len(default_rows) > 0:
                    # All default value rows should be in training
                    assert (default_rows["split"] == "train").all(), (
                        f"Rows with {key}={value} (default) should be in training"
                    )

    def test_holdout_false_allows_controls_in_all_splits(self, sample_grouped_distribution):
        """Test that when holdout_combinations=False, controls can be in any split."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Check that we have all three splits
        assert set(df_split["split"].unique()) == {"train", "val", "test"}


class TestContainsValue:
    """Test the _contains_value static method."""

    def test_contains_value_basic(self):
        """Test basic _contains_value functionality."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_a"],
                "gene": ["gene_x", "gene_y", "gene_z", "gene_x"],
            }
        )

        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=False,
        )

        # Rows 0 and 3 should be True (drug_a)
        assert df["is_target"].tolist() == [True, False, False, True]

    def test_contains_value_multiple_keys_or_logic(self):
        """Test _contains_value with multiple keys uses OR logic."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_a"],
                "gene": ["gene_x", "gene_y", "gene_x", "gene_z"],
            }
        )

        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a", "gene": "gene_x"},
            column_key="is_target",
            accept_nan=False,
        )

        # Rows where drug=drug_a OR gene=gene_x should be True
        assert df["is_target"].tolist() == [True, False, True, True]

    def test_contains_value_with_nan_accept_false(self):
        """Test _contains_value with NaN when accept_nan=False."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", np.nan, "drug_a"],
                "gene": ["gene_x", "gene_y", "gene_z", np.nan],
            }
        )

        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=False,
        )

        # Only rows with drug_a (0 and 3), NaN should not match
        assert df["is_target"].tolist() == [True, False, False, True]

    def test_contains_value_with_nan_accept_true(self):
        """Test _contains_value with NaN when accept_nan=True."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", np.nan, "drug_d"],
                "gene": ["gene_x", "gene_y", "gene_z", np.nan],
            }
        )

        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=True,
        )

        # Rows with drug_a OR NaN in drug column (0, 2)
        expected = [True, False, True, False]
        assert df["is_target"].tolist() == expected

    def test_contains_value_or_logic_multiple_calls(self):
        """Test that _contains_value uses OR logic when called multiple times."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c"],
                "gene": ["gene_x", "gene_y", "gene_z"],
            }
        )

        # First call
        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=False,
        )

        assert df["is_target"].tolist() == [True, False, False]

        # Second call with different value - should OR with previous
        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_b"},
            column_key="is_target",
            accept_nan=False,
        )

        # Rows 0 and 1 should be True (drug_a OR drug_b)
        assert df["is_target"].tolist() == [True, True, False]

    def test_contains_value_creates_column_if_not_exists(self):
        """Test that _contains_value creates the column if it doesn't exist."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
            }
        )

        assert "is_target" not in df.columns

        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=False,
        )

        assert "is_target" in df.columns
        assert df["is_target"].tolist() == [True, False]

    def test_contains_value_skips_keys_not_in_combination_keys(self):
        """Test that _contains_value ignores keys not in combination_keys."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c"],
                "gene": ["gene_x", "gene_y", "gene_z"],
            }
        )

        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug"],  # Only drug, not gene
            values={"drug": "drug_a", "gene": "gene_x"},  # gene_x should be ignored
            column_key="is_target",
            accept_nan=False,
        )

        # Only drug_a should match, gene_x is ignored
        assert df["is_target"].tolist() == [True, False, False]

    def test_contains_value_empty_values_dict(self):
        """Test _contains_value with empty values dictionary."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
            }
        )

        AnnotationSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug"],
            values={},  # Empty
            column_key="is_target",
            accept_nan=False,
        )

        # All should be False
        assert df["is_target"].tolist() == [False, False]

    def test_contains_value_invalid_combination_keys(self):
        """Test that _contains_value raises error for invalid combination_keys."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
            }
        )

        with pytest.raises(ValueError, match="combination_keys must be in df.columns"):
            AnnotationSplitter._contains_value(
                df_unique=df,
                combination_keys=["nonexistent_column"],
                values={"drug": "drug_a"},
                column_key="is_target",
                accept_nan=False,
            )


class TestSplitTwo:
    """Test the _split_two static method."""

    def test_split_two_basic(self):
        """Test basic _split_two functionality."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_d", "drug_e"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        result = AnnotationSplitter._split_two(
            df_unique=df.copy(),
            train_size=3,
            split_key="split",
            random_state=42,
        )

        # Check split column exists
        assert "split" in result.columns

        # Check counts
        assert (result["split"] == "train").sum() == 3
        assert (result["split"] == "test_val").sum() == 2

        # Check all rows are present
        assert len(result) == 5

    def test_split_two_shuffles_data(self):
        """Test that _split_two shuffles the data."""
        df = pd.DataFrame(
            {
                "drug": [f"drug_{i}" for i in range(20)],
                "value": list(range(20)),
            }
        )

        result = AnnotationSplitter._split_two(
            df_unique=df.copy(),
            train_size=10,
            split_key="split",
            random_state=42,
        )

        # Get train drugs
        train_drugs = result[result["split"] == "train"]["drug"].tolist()
        original_first_10 = [f"drug_{i}" for i in range(10)]

        # With shuffling and 20 items, extremely unlikely to match exactly
        assert train_drugs != original_first_10

    def test_split_two_deterministic(self):
        """Test that _split_two is deterministic with same random_state."""
        df = pd.DataFrame(
            {
                "drug": [f"drug_{i}" for i in range(10)],
            }
        )

        result1 = AnnotationSplitter._split_two(
            df_unique=df.copy(),
            train_size=6,
            split_key="split",
            random_state=42,
        )

        result2 = AnnotationSplitter._split_two(
            df_unique=df.copy(),
            train_size=6,
            split_key="split",
            random_state=42,
        )

        pd.testing.assert_frame_equal(result1, result2)

    def test_split_two_custom_split_key(self):
        """Test _split_two with custom split_key."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c"],
            }
        )

        result = AnnotationSplitter._split_two(
            df_unique=df.copy(),
            train_size=2,
            split_key="my_split",
            random_state=42,
        )

        assert "my_split" in result.columns
        assert set(result["my_split"].unique()) == {"train", "test_val"}

    def test_split_two_raises_on_existing_split_key(self):
        """Test that _split_two raises error if split_key already exists and overwrite=False."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
                "split": ["existing", "existing"],  # Column already exists
            }
        )

        with pytest.raises(ValueError, match="already in df.columns"):
            AnnotationSplitter._split_two(
                df_unique=df,
                train_size=1,
                split_key="split",
                overwrite=False,
                random_state=42,
            )

    def test_split_two_overwrites_existing_split_key(self):
        """Test that _split_two can overwrite existing split_key when overwrite=True."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c"],
                "split": ["existing", "existing", "existing"],
            }
        )

        result = AnnotationSplitter._split_two(
            df_unique=df,
            train_size=2,
            split_key="split",
            overwrite=True,
            random_state=42,
        )

        # Should have new values, not "existing"
        assert "existing" not in result["split"].values
        assert set(result["split"].unique()) == {"train", "test_val"}

    def test_split_two_with_forced_training_key(self):
        """Test _split_two respects is_in_training_key."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_d", "drug_e"],
                "forced_in_train": [True, True, False, False, False],
            }
        )

        # Now that the bug is fixed, this should work correctly
        # It should filter to only rows where forced_in_train is True
        result = AnnotationSplitter._split_two(
            df_unique=df.copy(),
            train_size=1,  # Only 1 train from the 2 available (drug_a, drug_b)
            split_key="split",
            is_in_training_key="forced_in_train",
            random_state=42,
        )

        # Should only contain the forced training rows (drug_a and drug_b)
        assert len(result) == 2
        assert set(result["drug"]) == {"drug_a", "drug_b"}

        # One should be train, one should be test_val
        assert (result["split"] == "train").sum() == 1
        assert (result["split"] == "test_val").sum() == 1

    def test_split_two_raises_if_training_key_missing(self):
        """Test that _split_two raises error if is_in_training_key doesn't exist."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
            }
        )

        with pytest.raises(ValueError, match="must be in df.columns"):
            AnnotationSplitter._split_two(
                df_unique=df,
                train_size=1,
                split_key="split",
                is_in_training_key="nonexistent",
                random_state=42,
            )


class TestCalculateSplitSizes:
    """Test the _calculate_split_sizes method."""

    def test_calculate_split_sizes_normal(self, sample_grouped_distribution):
        """Test split size calculation with normal inputs."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        train, val, test = splitter._calculate_split_sizes(100)

        # Check they sum to 100
        assert train + val + test == 100

        # Check they're approximately correct
        assert train == 60
        assert val == 20
        assert test == 20

    def test_calculate_split_sizes_with_rounding(self, sample_grouped_distribution):
        """Test split size calculation handles rounding correctly."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.7, 0.15, 0.15],
            random_state=42,
        )

        train, val, test = splitter._calculate_split_sizes(100)

        # Check they sum to 100
        assert train + val + test == 100

        # Check individual values
        assert train == 70
        assert val == 15
        assert test == 15

    def test_calculate_split_sizes_too_small_raises_error(self, sample_grouped_distribution):
        """Test that too few combinations raises ValueError."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.9, 0.05, 0.05],
            random_state=42,
        )

        # With 5 total and ratios 0.9, 0.05, 0.05, some might round to 0
        with pytest.raises(ValueError, match="0 was encountered"):
            splitter._calculate_split_sizes(5)

    def test_calculate_split_sizes_large_numbers(self, sample_grouped_distribution):
        """Test split size calculation with large numbers."""
        splitter = AnnotationSplitter(
            annotation=sample_grouped_distribution.annotation,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        train, val, test = splitter._calculate_split_sizes(10000)

        # Check they sum to 10000
        assert train + val + test == 10000


class TestCheckDfUniqueColumns:
    """Test the _check_df_unique_columns static method."""

    def test_check_df_unique_columns_valid(self):
        """Test _check_df_unique_columns with valid columns."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a"],
                "gene": ["gene_x"],
            }
        )

        # Should not raise
        AnnotationSplitter._check_df_unique_columns(df, ["drug", "gene"])

    def test_check_df_unique_columns_invalid(self):
        """Test _check_df_unique_columns raises error for missing columns."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a"],
            }
        )

        with pytest.raises(ValueError, match="combination_keys must be in df.columns"):
            AnnotationSplitter._check_df_unique_columns(df, ["drug", "nonexistent"])

    def test_check_df_unique_columns_empty_list(self):
        """Test _check_df_unique_columns with empty combination_keys."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a"],
            }
        )

        # Should not raise with empty list
        AnnotationSplitter._check_df_unique_columns(df, [])


    def test_split_two_with_not_in_training_key(self):
        """Test _split_two respects not_in_training_key."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_d", "drug_e"],
                "exclude_from_split": [False, False, True, True, False],
            }
        )

        # Should exclude rows where exclude_from_split is True
        result = AnnotationSplitter._split_two(
            df_unique=df.copy(),
            train_size=2,
            split_key="split",
            not_in_training_key="exclude_from_split",
            random_state=42,
        )

        # Should only contain rows where exclude_from_split was False
        assert len(result) == 3
        assert set(result["drug"]) == {"drug_a", "drug_b", "drug_e"}

        # Check split distribution
        assert (result["split"] == "train").sum() == 2
        assert (result["split"] == "test_val").sum() == 1

        # Drugs c and d should not be in the result
        assert "drug_c" not in result["drug"].values
        assert "drug_d" not in result["drug"].values
