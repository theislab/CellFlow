import anndata as ad
import numpy as np
import pytest


class TestPCA:
    def test_centered_pca(self, adata_pca: ad.AnnData):
        import scaleflow

        scaleflow.pp.centered_pca(adata_pca, n_comps=50, copy=False)
        assert "X_pca" in adata_pca.obsm
        assert "PCs" in adata_pca.varm
        assert "X_mean" in adata_pca.varm
        assert "X_centered" in adata_pca.layers
        assert "variance_ratio" in adata_pca.uns["pca"]
        assert "variance" in adata_pca.uns["pca"]

    @pytest.mark.parametrize("layers_key_added", ["X_recon", "X_rec"])
    def test_reconstruct_pca(self, adata_pca: ad.AnnData, layers_key_added):
        import scaleflow

        scaleflow.pp.centered_pca(adata_pca, n_comps=50, copy=False)
        scaleflow.pp.reconstruct_pca(
            adata_pca,
            ref_adata=adata_pca,
            use_rep="X_pca",
            layers_key_added=layers_key_added,
            copy=False,
        )
        assert layers_key_added in adata_pca.layers
        assert np.allclose(
            adata_pca.layers[layers_key_added],
            adata_pca.X.toarray(),
            atol=1e-6,
            rtol=0.0,
        )

    def test_reconstruct_pca_with_array_input(self, adata_pca: ad.AnnData):
        import scaleflow

        scaleflow.pp.centered_pca(adata_pca, n_comps=50, copy=False)
        scaleflow.pp.reconstruct_pca(adata_pca, ref_means=adata_pca.varm["X_mean"], ref_pcs=adata_pca.varm["PCs"])
        assert "X_recon" in adata_pca.layers

    @pytest.mark.parametrize("obsm_key_added", ["X_pca", "X_pca_projected"])
    def test_project_pca(self, adata_pca: ad.AnnData, obsm_key_added):
        import scaleflow

        scaleflow.pp.centered_pca(adata_pca, n_comps=50, copy=False)
        adata_pca_project = scaleflow.pp.project_pca(
            adata_pca, ref_adata=adata_pca, obsm_key_added=obsm_key_added, copy=True
        )
        assert obsm_key_added in adata_pca_project.obsm
        assert np.allclose(adata_pca_project.obsm[obsm_key_added], adata_pca.obsm["X_pca"])
