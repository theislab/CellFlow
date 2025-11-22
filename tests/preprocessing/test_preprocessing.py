import anndata as ad
import numpy as np
import pytest


class TestPreprocessing:
    @pytest.mark.parametrize(
        "compound_key_and_type",
        [
            (["compound_name"], "name"),
            (["compound_cid"], "cid"),
            (["compound_name", "compound2_name"], "name"),
        ],
    )
    def test_annotate_compounds(self, adata_with_compounds: ad.AnnData, compound_key_and_type):
        import scaleflow

        try:
            scaleflow.pp.annotate_compounds(
                adata_with_compounds,
                compound_keys=compound_key_and_type[0],
                query_id_type=compound_key_and_type[1],
                copy=False,
            )
        except Exception as e:
            if "ServerBusy" in str(e):
                pytest.skip("Skipped test due to PubChem server being busy.")
            else:
                raise

        for compound_key in compound_key_and_type[0]:
            assert f"{compound_key}_pubchem_name" in adata_with_compounds.obs
            assert f"{compound_key}_pubchem_ID" in adata_with_compounds.obs
            assert f"{compound_key}_smiles" in adata_with_compounds.obs

    @pytest.mark.parametrize("n_bits", [512, 1024])
    @pytest.mark.parametrize(
        "compound_and_smiles_keys",
        [
            ("compound_name", "compound_smiles"),
            (
                ["compound_name", "compound2_name"],
                ["compound_smiles", "compound2_smiles"],
            ),
        ],
    )
    def test_get_molecular_fingerprints(self, adata_with_compounds: ad.AnnData, n_bits, compound_and_smiles_keys):
        import scaleflow

        uns_key_added = "compound_fingerprints"

        scaleflow.pp.get_molecular_fingerprints(
            adata_with_compounds,
            compound_keys=compound_and_smiles_keys[0],
            smiles_keys=compound_and_smiles_keys[1],
            uns_key_added=uns_key_added,
            n_bits=n_bits,
            copy=False,
        )

        assert uns_key_added in adata_with_compounds.uns
        assert next(iter(adata_with_compounds.uns[uns_key_added].values())).shape[0] == n_bits
        expected_compounds = adata_with_compounds.obs[compound_and_smiles_keys[0]].values.flatten().tolist()

        assert np.all([c in adata_with_compounds.uns[uns_key_added] for c in expected_compounds])

    @pytest.mark.parametrize("uns_key_added", ["compounds", "compounds_onehot"])
    @pytest.mark.parametrize("exclude_values", [None, "GW0742"])
    def test_encode_onehot(self, adata_with_compounds: ad.AnnData, uns_key_added, exclude_values):
        import scaleflow

        scaleflow.pp.encode_onehot(
            adata_with_compounds,
            covariate_keys="compound_name",
            uns_key_added=uns_key_added,
            exclude_values=exclude_values,
            copy=False,
        )

        if exclude_values is None:
            expected_compounds = adata_with_compounds.obs["compound_name"].unique()
        else:
            expected_compounds = np.setdiff1d(adata_with_compounds.obs["compound_name"].unique(), exclude_values)
        assert uns_key_added in adata_with_compounds.uns
        assert len(adata_with_compounds.uns[uns_key_added]) == len(expected_compounds)
        assert adata_with_compounds.uns[uns_key_added][expected_compounds[0]].shape[0] == len(expected_compounds)
        assert np.all([c in adata_with_compounds.uns[uns_key_added] for c in expected_compounds])
