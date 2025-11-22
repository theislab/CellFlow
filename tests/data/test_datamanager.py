import numpy as np
import pytest

from scaleflow.data import AnnDataLocation, DataManager, GroupedDistribution


class TestDataManagerBasic:
    """Test basic DataManager functionality."""

    def test_prepare_data_basic(self, adata_test):
        """Test that prepare_data works and returns correct structure."""
        adl = AnnDataLocation()

        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_test)

        assert isinstance(gd, GroupedDistribution)

        # src_dist_idx	tgt_dist_idx	cell_line	drug	gene
        #   0	0	cell_line_A	control	gene_A
        #   0	1	cell_line_A	control	gene_B
        #   0	2	cell_line_A	drug_A	control
        #   0	3	cell_line_A	drug_A	gene_A
        #   0	4	cell_line_A	drug_A	gene_B
        #   0	5	cell_line_A	drug_B	control
        #   0	6	cell_line_A	drug_B	gene_A
        #   0	7	cell_line_A	drug_B	gene_B
        #   1	8	cell_line_B	control	gene_A
        #   1	9	cell_line_B	control	gene_B
        #   1	10	cell_line_B	drug_A	control
        #   1	11	cell_line_B	drug_A	gene_A
        #   1	12	cell_line_B	drug_A	gene_B
        #   1	13	cell_line_B	drug_B	control
        #   1	14	cell_line_B	drug_B	gene_A
        #   1	15	cell_line_B	drug_B	gene_B

        expected_src_data = {0: ("cell_line_A",), 1: ("cell_line_B",)}
        expected_tgt_data = {
            0: ("drug_A", "control"),
            1: ("drug_B", "control"),
            2: ("gene_A", "control"),
            3: ("gene_B", "control"),
            4: ("drug_A", "gene_A"),
            5: ("drug_A", "gene_B"),
            6: ("drug_B", "gene_A"),
            7: ("drug_B", "gene_B"),
            8: ("drug_A", "control"),
            9: ("drug_B", "control"),
            10: ("gene_A", "control"),
            11: ("gene_B", "control"),
            12: ("drug_A", "gene_A"),
            13: ("drug_A", "gene_B"),
            14: ("drug_B", "gene_A"),
            15: ("drug_B", "gene_B"),
        }
        expected_mapping = {
            0: {0, 1, 2, 3, 4, 5, 6, 7},
            1: {8, 9, 10, 11, 12, 13, 14, 15},
        }

        assert len(gd.data.src_data) == len(expected_src_data)
        assert len(gd.data.tgt_data) == len(expected_tgt_data)

        # Test target mapping correctness
        # Verify that src_to_tgt_dist_map exists for each source
        assert len(gd.data.src_to_tgt_dist_map) == len(expected_src_data)

        # sum of the values in src_to_tgt_dist_map should be equal to the number of target distributions
        assert sum(len(v) for v in gd.data.src_to_tgt_dist_map.values()) == len(expected_tgt_data)

        # Each source should have at least one target
        for src_idx, tgt_indices in gd.data.src_to_tgt_dist_map.items():
            assert len(tgt_indices) > 0, f"Source {src_idx} has no targets"
            # All target indices should exist in tgt_data
            for tgt_idx in tgt_indices:
                assert tgt_idx in gd.data.tgt_data, f"Target {tgt_idx} not in tgt_data"

        # Verify that targets are correctly mapped to their source cell_lines
        # using the src_tgt_dist_df
        src_tgt_df = gd.annotation.src_tgt_dist_df

        # For each target, verify it belongs to the correct source
        for _, row in src_tgt_df.iterrows():
            src_idx = row["src_dist_idx"]
            tgt_idx = row["tgt_dist_idx"]
            # Target should be in the source's target list
            assert tgt_idx in gd.data.src_to_tgt_dist_map[src_idx], (
                f"Target {tgt_idx} not found in source {src_idx}'s mapping"
            )
            assert tgt_idx in expected_mapping[src_idx], f"Target {tgt_idx} not found in source {src_idx}'s mapping"

    def test_ordering_reconstruction_after_shuffle(self, adata_test):
        """Test that we can reconstruct original ordering after shuffling."""
        # Store original order information
        original_index = adata_test.obs.index.to_numpy().copy()
        original_X_pca = adata_test.obsm["X_pca"].copy()

        # Shuffle the adata
        shuffle_idx = np.random.permutation(len(adata_test))
        adata_shuffled = adata_test[shuffle_idx].copy()

        # Verify it's actually shuffled (should not be identical for reasonable dataset sizes)
        assert not np.array_equal(adata_shuffled.obs.index.to_numpy(), original_index), "Data should be shuffled"

        # Create DataManager and prepare data
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_shuffled)

        # Test 1: old_obs_index should map from sorted order back to shuffled AnnData index
        assert len(gd.annotation.old_obs_index) == len(adata_shuffled)
        assert np.all(np.isin(gd.annotation.old_obs_index, adata_shuffled.obs.index.to_numpy())), (
            "old_obs_index should contain valid indices from shuffled adata"
        )

        # Verify: For each position in the sorted data, the old_obs_index tells us
        # which original (shuffled adata) index it came from
        for _, old_idx in enumerate(gd.annotation.old_obs_index):
            # Find the corresponding cell in the shuffled adata
            shuffled_pos = np.where(adata_shuffled.obs.index == old_idx)[0][0]

            # The first element of X_pca contains the original cell index from adata_test
            # (this was set up in the fixture)
            original_cell_id = adata_shuffled.obsm["X_pca"][shuffled_pos, 0]

            # This should match the original unshuffled data
            assert original_cell_id == original_X_pca[np.where(original_index == old_idx)[0][0], 0]

        # Test 3: Verify we can fully reconstruct the mapping
        # Create inverse mapping: from old_obs_index position -> shuffled adata position
        old_idx_to_shuffled_pos = {
            old_idx: np.where(adata_shuffled.obs.index == old_idx)[0][0] for old_idx in gd.annotation.old_obs_index
        }

        # This should cover all cells in the shuffled adata
        assert len(old_idx_to_shuffled_pos) == len(adata_shuffled)

