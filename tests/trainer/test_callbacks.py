import anndata as ad
import jax.numpy as jnp
import pytest


class TestCallbacks:
    @pytest.mark.parametrize("metrics", [["r_squared"]])
    def test_pca_reconstruction(self, adata_pca: ad.AnnData, metrics):
        from cellflow.training import PCADecodedMetrics

        decoded_metrics_callback = PCADecodedMetrics(
            metrics=metrics,
            ref_adata=adata_pca,
        )

        reconstruction = decoded_metrics_callback.reconstruct_data(adata_pca.obsm["X_pca"])
        assert reconstruction.shape == adata_pca.X.shape
        assert jnp.allclose(reconstruction, adata_pca.layers["counts"], atol=1e-5)
