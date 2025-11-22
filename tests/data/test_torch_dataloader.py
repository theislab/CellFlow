import scaleflow
from scaleflow.data import TorchCombinedTrainSampler


class TestTorchDataloader:
    def test_torch_dataloader_shapes(
        self,
        adata_perturbation,
        tmp_path,
    ):
        solver = "otfm"
        sample_rep = "X"
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1", "drug2"]}
        perturbation_covariate_reps = {"drug": "drug"}
        batch_size = 18

        cf = scaleflow.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
        )
        assert cf.train_data is not None
        assert hasattr(cf, "_data_dim")
        cf.train_data.write_zarr(tmp_path / "train_data1.zarr")
        cf.train_data.write_zarr(tmp_path / "train_data2.zarr")
        cf.train_data.write_zarr(tmp_path / "train_data3.zarr")

        combined_dataloader = TorchCombinedTrainSampler.combine_zarr_training_samplers(
            data_paths=[
                tmp_path / "train_data1.zarr",
                tmp_path / "train_data2.zarr",
                tmp_path / "train_data3.zarr",
            ],
            batch_size=batch_size,
            num_workers=2,
            weights=[0.3, 0.3, 0.4],
            seed=42,
            dataset_names=["train_data1", "train_data2", "train_data3"],
        )
        iter_dl = iter(combined_dataloader)
        batch = next(iter_dl)
        assert "dataset_name" in batch
        assert batch["dataset_name"] in ["train_data1", "train_data2", "train_data3"]
        assert "src_cell_data" in batch
        assert "tgt_cell_data" in batch
        assert "condition" in batch
        dim = adata_perturbation.shape[1]
        assert batch["src_cell_data"].shape == (batch_size, dim)
        assert batch["tgt_cell_data"].shape == (batch_size, dim)
        assert "drug" in batch["condition"]
        drug_dim = adata_perturbation.uns["drug"]["drug_a"].shape[0]
        assert batch["condition"]["drug"].shape == (1, len(perturbation_covariates["drug"]), drug_dim)
