import anndata as ad
import pytest

from cellflow.datasets import _load_dataset


@pytest.mark.internet
def test_load_dataset_from_hf():
    adata = _load_dataset(filename="test.h5ad")

    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (5, 3)
    assert list(adata.obs.columns) == ["cell_type"]
    assert list(adata.var_names) == ["gene_0", "gene_1", "gene_2"]
