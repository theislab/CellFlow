import anndata as ad
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from cellflow.data._data import TrainingData
from cellflow.data._dataloader import TrainSampler, ValidationSampler


@pytest.fixture
def dataloader():
    class DataLoader:
        n_conditions = 10

        def sample(self, rng):
            return {
                "src_cell_data": jnp.ones((10, 5)) * 10,
                "tgt_cell_data": jnp.ones((10, 5)),
                "condition": {"pert1": jnp.ones((1, 2, 3))},
            }

    return DataLoader()


@pytest.fixture
def validdata():
    class ValidData:
        n_conditions = 10

        def __init__(self):
            self.cell_data = jnp.ones((10, 5))
            self.condition_data = {"pert1": jnp.ones((1, 2, 3))}
            self.n_conditions_on_log_iteration = 1
            self.n_conditions_on_train_end = 1
            self.max_combination_length = 2
            self.control_to_perturbation = {0: [0]}
            self.n_perturbations = 1
            self.split_covariates_mask = jnp.zeros(
                len(self.cell_data),
            )
            self.perturbation_covariates_mask = jnp.zeros(
                len(self.cell_data),
            )
            self.perturbation_idx_to_covariates = {0: np.array(["my_pert"])}
            self.perturbation_idx_to_id = {0: "my_naming_of_pert"}

    return {"val": ValidData()}


@pytest.fixture()
def valid_loader(validdata):
    return {k: ValidationSampler(v) for k, v in validdata.items()}


@pytest.fixture
def big_validdata():
    class ValidDataToSubsample:
        def __init__(self):
            N_SOURCE = 10
            N_COND_TARGET = 5
            self.tgt_data = {}
            self.condition_data = {}
            self.src_data = {i: jnp.ones((10, 5)) * 10 for i in range(N_SOURCE)}
            for i in range(N_SOURCE):
                self.tgt_data[i] = {i * N_COND_TARGET + j: jnp.ones((10, 5)) for j in range(N_COND_TARGET)}
                for j in range(N_COND_TARGET):
                    self.condition_data[i * N_COND_TARGET + j] = {"pert1": jnp.ones((i + j, i + j * 2, 2 * i + j))}
            self.n_conditions_on_log_iteration = 4
            self.n_conditions_on_train_end = 16
            self.max_combination_length = 2

    return {"big_val": ValidDataToSubsample()}


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
    adata.varm["X_mean"] = adata.X.mean(axis=0).T
    adata.layers["counts"] = adata.X
    adata.X = csr_matrix(adata.X - adata.varm["X_mean"])
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
