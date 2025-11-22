import pytest
from scvi.data import synthetic_iid

from scaleflow.external import CFJaxSCVI


class TestCFJaxSCVI:
    @pytest.mark.parametrize("gene_likelihood", ["nb", "poisson", "normal"])
    def test_jax_scvi(self, gene_likelihood: str, n_latent=5):
        adata = synthetic_iid()
        CFJaxSCVI.setup_anndata(
            adata,
            batch_key="batch",
        )
        model = CFJaxSCVI(adata, n_latent=n_latent, gene_likelihood=gene_likelihood)
        model.train(2, train_size=0.5, check_val_every_n_epoch=1)
        model.get_latent_representation()

        model = CFJaxSCVI(adata, n_latent=n_latent, gene_likelihood=gene_likelihood)
        model.train(1, train_size=0.5)
        z1 = model.get_latent_representation(give_mean=True, n_samples=1)
        assert z1.ndim == 2
        z2 = model.get_latent_representation(give_mean=False, n_samples=15)
        assert z2.ndim == 3
        assert z2.shape[0] == 15

        adata.obsm["X_scVI"] = z1
        out = model.get_reconstructed_expression(adata)
        assert out.shape == adata.X.shape
