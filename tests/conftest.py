import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scaleflow.data import AnnDataLocation, DataManager, GroupedDistribution


@pytest.fixture()
def adata_perturbation() -> ad.AnnData:
    n_obs = 500
    n_vars = 50
    n_pca = 10

    X_data = np.random.rand(n_obs, n_vars)

    my_counts = np.random.rand(n_obs, n_vars)

    X_pca = np.random.rand(n_obs, n_pca)

    cell_lines = np.random.choice(["cell_line_a", "cell_line_b", "cell_line_c"], n_obs)
    dosages = np.random.choice([10.0, 100.0, 1000.0], n_obs)
    drugs = ["drug_a", "drug_b", "drug_c"]
    drug1 = np.random.choice(drugs, n_obs)
    drug2 = np.random.choice(drugs, n_obs)
    drug3 = np.random.choice(drugs, n_obs)
    dosages_a = np.random.choice([10.0, 100.0, 1000.0], n_obs)
    dosages_b = np.random.choice([10.0, 100.0, 1000.0], n_obs)
    dosages_c = np.random.choice([10.0, 100.0, 1000.0], n_obs)

    obs_data = pd.DataFrame(
        {
            "cell_type": cell_lines,
            "dosage": dosages,
            "drug1": drug1,
            "drug2": drug2,
            "drug3": drug3,
            "dosage_a": dosages_a,
            "dosage_b": dosages_b,
            "dosage_c": dosages_c,
        }
    )

    # Create an AnnData object
    adata = ad.AnnData(X=X_data, obs=obs_data)

    # Add the random data to .layers and .obsm
    adata.layers["my_counts"] = my_counts
    adata.obsm["X_pca"] = X_pca

    control_idcs = np.random.choice(n_obs, n_obs // 10, replace=False)
    for col in ["drug1", "drug2", "drug3"]:
        adata.obs.loc[[str(idx) for idx in control_idcs], col] = "control"

    adata.obs["drug_a"] = (
        (adata.obs["drug1"] == "drug_a") | (adata.obs["drug2"] == "drug_a") | (adata.obs["drug3"] == "drug_a")
    )

    for col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype("category")

    adata.obs["drug_b"] = (
        (adata.obs["drug1"] == "drug_b") | (adata.obs["drug2"] == "drug_b") | (adata.obs["drug3"] == "drug_b")
    )
    adata.obs["drug_c"] = (
        (adata.obs["drug1"] == "drug_c") | (adata.obs["drug2"] == "drug_c") | (adata.obs["drug3"] == "drug_c")
    )
    adata.obs["control"] = adata.obs["drug1"] == "control"

    drug_emb = {}
    for drug in adata.obs["drug1"].cat.categories:
        drug_emb[drug] = np.random.randn(5, 1)
    adata.uns["drug"] = drug_emb
    cell_type_emb = {}
    for cell_type in adata.obs["cell_type"].cat.categories:
        cell_type_emb[cell_type] = np.random.randn(3, 1)
    adata.uns["cell_type"] = cell_type_emb

    return adata


@pytest.fixture()
def adata_perturbation_with_nulls(adata_perturbation: ad.AnnData) -> ad.AnnData:
    adata = adata_perturbation.copy()
    del adata.obs["drug1"]
    del adata.obs["drug2"]
    del adata.obs["drug3"]
    n_obs = adata.n_obs
    drugs = ["drug_a", "drug_b", "drug_c", "control", "no_drug"]
    drug1 = np.random.choice(drugs, n_obs)
    drug2 = np.random.choice(drugs, n_obs)
    drug3 = np.random.choice(drugs, n_obs)
    adata.obs["drug1"] = drug1
    adata.obs["drug2"] = drug2
    adata.obs["drug3"] = drug3
    adata.obs["drug1"] = adata.obs["drug1"].astype("category")
    adata.obs["drug2"] = adata.obs["drug2"].astype("category")
    adata.obs["drug3"] = adata.obs["drug3"].astype("category")

    return adata


@pytest.fixture()
def adata_pca() -> ad.AnnData:
    import scanpy as sc
    from scipy.sparse import csr_matrix

    n_obs = 10
    n_vars = 50
    n_pca = 10

    X_data = np.random.rand(n_obs, n_vars)
    adata = ad.AnnData(X=X_data)

    # Add the random data to .layers and .obsm
    adata.varm["X_mean"] = adata.X.mean(axis=0)
    adata.layers["counts"] = adata.X
    adata.X = csr_matrix(adata.X - adata.varm["X_mean"].T)
    sc.pp.pca(adata, zero_center=False, n_comps=n_pca)

    return adata


@pytest.fixture()
def adata_with_compounds() -> ad.AnnData:
    n_obs = 10
    n_vars = 50
    compound_names = np.array(["AZD1390", "Dabrafenib Mesylate", "GW0742"])
    compound_cids = np.array([126689157, 44516822, 9934458])
    compound_smiles = np.array(
        [
            "CC(C)N1C2=C(C=NC3=CC(=C(C=C32)C4=CN=C(C=C4)OCCCN5CCCCC5)F)N(C1=O)C",
            "CC(C)(C)C1=NC(=C(S1)C2=NC(=NC=C2)N)C3=C(C(=CC=C3)NS(=O)(=O)C4=C(C=CC=C4F)F)F.CS(=O)(=O)O",
            "CC1=C(C=CC(=C1)SCC2=C(N=C(S2)C3=CC(=C(C=C3)C(F)(F)F)F)C)OCC(=O)O",
        ]
    )
    compound_idcs = np.random.choice(len(compound_names), n_obs)

    X_data = np.random.rand(n_obs, n_vars)
    adata = ad.AnnData(X=X_data)
    adata.obs["compound_name"] = compound_names[compound_idcs]
    adata.obs["compound_cid"] = compound_cids[compound_idcs]
    adata.obs["compound_smiles"] = compound_smiles[compound_idcs]
    adata.obs["compound2_name"] = compound_names[compound_idcs]
    adata.obs["compound2_cid"] = compound_cids[compound_idcs]
    adata.obs["compound2_smiles"] = compound_smiles[compound_idcs]

    return adata


@pytest.fixture
def metrics_data():
    data = {}
    data["x_test"] = {
        "Alvespimycin+Pirarubicin": np.random.rand(50, 10),
        "Dacinostat+Danusertib": np.random.rand(50, 10),
    }
    data["y_test"] = {
        "Alvespimycin+Pirarubicin": np.random.rand(20, 10),
        "Dacinostat+Danusertib": np.random.rand(20, 10),
    }
    return data


@pytest.fixture
def adata_test() -> ad.AnnData:
    n_drugs = 10
    n_genes = 5
    n_cell_lines = 3
    n_plates = 3
    n_days = 3
    n_batches = 3
    drugs = ["control"] + [f"drug_{i}" for i in range(n_drugs)]
    genes = ["control"] + [f"gene_{i}" for i in range(n_genes)]
    cell_lines = [f"cell_line_{i}" for i in range(n_cell_lines)]
    batches = [f"batch_{i}" for i in range(n_batches)]
    plates = [f"plate_{i}" for i in range(n_plates)]
    days = [f"day_{i}" for i in range(n_days)]
    doses = [1.0, 10.0, 100.0]

    rows = []
    for drug in drugs:
        for gene in genes:
            for cell_line in cell_lines:
                for batch in batches:
                    for plate in plates:
                        for day in days:
                            if drug != "control":
                                for dose in doses:
                                    rows.append(
                                        {
                                            "drug": drug,
                                            "gene": gene,
                                            "cell_line": cell_line,
                                            "batch": batch,
                                            "plate": plate,
                                            "day": day,
                                            "dose": dose,
                                            "control": False,
                                        }
                                    )
                            else:
                                rows.append(
                                    {
                                        "drug": drug,
                                        "gene": gene,
                                        "cell_line": cell_line,
                                        "batch": batch,
                                        "plate": plate,
                                        "day": day,
                                        "dose": 0.0,
                                        "control": gene == "control" and drug == "control",
                                    }
                                )

    n_obs = len(rows)
    n_vars = 20
    n_pca = 10

    obs = pd.DataFrame(rows)

    # Convert to categorical
    for col in ["cell_line", "drug", "gene", "batch", "plate", "day"]:
        obs[col] = obs[col].astype("category")

    # Simple X matrix (not really used in tests, just needs to exist)
    X = np.random.randn(n_obs, n_vars).astype(np.float32)

    # X_pca: Put cell index at position [idx, 0] for easy tracing
    X_pca = np.zeros((n_obs, n_pca), dtype=np.float32)
    for i in range(n_obs):
        X_pca[i, 0] = float(i)  # Cell 0 has value 0, cell 1 has value 1, etc.

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = X_pca

    # Simple embeddings
    # one hot encoding

    adata.uns["cell_line_embeddings"] = dict(zip(cell_lines, np.eye(n_cell_lines), strict=True))

    adata.uns["drug_embeddings"] = dict(
        zip(drugs, np.concatenate([np.zeros((1, n_drugs)), np.eye(n_drugs)], axis=0), strict=True)
    )

    adata.uns["gene_embeddings"] = dict(
        zip(genes, np.concatenate([np.zeros((1, n_genes)), np.eye(n_genes)], axis=0), strict=True)
    )

    return adata


@pytest.fixture
def sample_grouped_distribution(adata_test: ad.AnnData) -> GroupedDistribution:
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
    return gd
