import os
from collections import Counter

import anndata as ad
import pandas as pd
import pytest
import torch

from cellflow.preprocessing._gene_emb import get_esm_embedding

IS_PROT_CODING = Counter(["ENSG00000139618", "ENSG00000206450", "ENSG00000049192"])
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_artifacts/")


@pytest.fixture()
def adata_with_ko():
    adata = ad.AnnData(
        obs=pd.DataFrame(
            {
                "gene_target_1": [
                    "ENSG00000139618",
                    "ENSG00000139618",
                    "ENSG00000260123",
                    "ENSG00000049192",
                ],
                "gene_target_2": [
                    "ENSG00000049192",
                    "ENSG00000206450",
                    None,
                    None,
                ],
                "cell_id": ["cell1", "cell2", "cell3", "cell4"],
            }
        )
    )
    return adata


@pytest.fixture()
def adata_test_legacy():
    adata = ad.AnnData(
        obs=pd.DataFrame(
            {
                "gene": [
                    "ENSG00000171105",
                    "ENSG00000169047",
                ],
            }
        )
    )
    return adata


class TestGeneEmb:
    @pytest.skip(reason="temporarily disabled")
    def test_embedding(self, adata_with_ko):
        adata = get_esm_embedding(adata_with_ko, gene_key="gene_target_", copy=True)
        metadata = adata.uns["gene_embedding_metadata"]
        assert Counter(metadata[metadata.is_protein_coding].gene_id.tolist()) == IS_PROT_CODING
        gene_with_prot_seq = metadata[metadata.protein_sequence.notnull()].gene_id.tolist()
        assert Counter(gene_with_prot_seq) == IS_PROT_CODING
        
    @pytest.skip(reason="temporarily disabled")
    def test_legacy_emb(self, adata_test_legacy):
        """Test if we can reproduce the original embeddings we used."""
        adata = get_esm_embedding(adata_test_legacy, gene_key="gene", copy=True)
        all_genes = adata.obs.gene.tolist()
        for gene in all_genes:
            emb = adata.uns["gene_embedding"][gene]
            fname = f"{gene}_emb.pt"
            legacy_emb = torch.load(os.path.join(ARTIFACTS_DIR, fname))
            legacy_emb = legacy_emb["mean_representations"][36]
            assert torch.allclose(emb, legacy_emb, atol=1e-2, rtol=1e-2)
