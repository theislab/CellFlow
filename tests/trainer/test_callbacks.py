import anndata as ad
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
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
        assert jnp.allclose(reconstruction, adata_pca.layers["counts"])

    @pytest.mark.parametrize("sparse_matrix", [True, False])
    @pytest.mark.parametrize("layers", [None, "test"])
    def test_pca_decoded_2(self, adata_pca: ad.AnnData, sparse_matrix, layers):
        from cellflow.solvers import OTFlowMatching
        from cellflow.training import PCADecodedMetrics2

        adata_gt = adata_pca.copy()
        adata_gt.obs["condition"] = np.random.choice(["A", "B"], size=adata_pca.shape[0])
        if not sparse_matrix:
            adata_gt.X = adata_gt.X.toarray()
        if layers is not None:
            adata_gt.layers[layers] = adata_gt.X.copy()

        decoded_metrics_callback = PCADecodedMetrics2(
            ref_adata=adata_pca,
            validation_adata={"test": adata_gt},
            metrics=["r_squared"],
            condition_id_key="condition",
            layers=layers,
        )

        valid_pred_data = {"test": {"A": np.random.random((2, 10)), "B": np.random.random((2, 10))}}

        res = decoded_metrics_callback.on_log_iteration({}, {}, valid_pred_data, OTFlowMatching)
        assert "pca_decoded_2_test_r_squared_mean" in res
        assert isinstance(res["pca_decoded_2_test_r_squared_mean"], float)

    @pytest.mark.parametrize("metrics", [["r_squared"]])
    def test_vae_reconstruction(self, metrics):
        from scvi.data import synthetic_iid

        from cellflow.external import CFJaxSCVI
        from cellflow.training import VAEDecodedMetrics

        adata = synthetic_iid()
        CFJaxSCVI.setup_anndata(
            adata,
            batch_key="batch",
        )
        model = CFJaxSCVI(adata, n_latent=2, gene_likelihood="normal")
        model.train(2, train_size=0.5, check_val_every_n_epoch=1)
        out = model.get_latent_representation(give_mean=True)

        vae_decoded_metrics_callback = VAEDecodedMetrics(
            vae=model,
            adata=adata,
            metrics=metrics,
        )

        dict_to_reconstruct = {"dummy": out}
        dict_adatas = jtu.tree_map(vae_decoded_metrics_callback._create_anndata, dict_to_reconstruct)
        reconstructed_arrs = jtu.tree_map(vae_decoded_metrics_callback.reconstruct_data, dict_adatas)
        assert reconstructed_arrs["dummy"].shape == adata.X.shape
