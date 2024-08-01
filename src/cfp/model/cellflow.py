import os
from collections.abc import Callable, Sequence
from typing import Any, Literal

import anndata as ad
import cloudpickle
import jax
import optax
from numpy.typing import ArrayLike
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from cfp.data.data import TrainingData, ValidationData
from cfp.data.dataloader import TrainSampler
from cfp.networks.velocity_field import ConditionalVelocityField
from cfp.solvers import genot, otfm
from cfp.training.trainer import CellFlowTrainer

__all__ = ["CellFlow"]


class CellFlow:
    """CellFlow model for perturbation prediction using Flow Matching.

    Args:
        adata: Anndata object.
        solver: Solver to use for training the model.

    """

    def __init__(self, adata: ad.AnnData, solver: Literal["otfm", "genot"] = "otfm"):

        self.adata = adata
        self.solver = solver
        self.dataloader: TrainSampler | None = None
        self.trainer: CellFlowTrainer | None = None
        self._validation_data: dict[str, ValidationData] = {}
        self._solver: otfm.OTFlowMatching | genot.GENOT | None = None
        self._condition_dim: int | None = None

    def prepare_data(
        self,
        sample_rep: str,
        control_key: str,
        perturbation_covariates: dict[str, Sequence[str]],
        perturbation_covariate_reps: dict[str, str] | None = None,
        sample_covariates: Sequence[str] | None = None,
        sample_covariate_reps: dict[str, str] | None = None,
        split_covariates: Sequence[str] | None = None,
        max_combination_length: int | None = None,
        null_value: float = 0.0,
    ) -> None:
        """Prepare dataloader for training from anndata object.

        Args:
            adata: An :class:`~anndata.AnnData` object.
            sample_rep: Key in `adata.obsm` where the sample representation is stored or "X" to use `adata.X`.
            control_key: Key of a boolean column in `adata.obs` that defines the control samples.
            perturbation_covariates: A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.obs`. The corresponding columns should be either boolean (presence/abscence of the perturbation) or numeric (concentration or magnitude of the perturbation). If multiple groups are provided, the first is interpreted as the primary perturbation and the others as covariates corresponding to these perturbations, e.g. `{"drug":("drugA", "drugB"), "time":("drugA_time", "drugB_time")}`.
            perturbation_covariate_reps: A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.uns` storing a dictionary with the representation of the covariates. E.g. `{"drug":"drug_embeddings"}` with `adata.uns["drug_embeddings"] = {"drugA": np.array, "drugB": np.array}`.
            sample_covariates: Keys in `adata.obs` indicating sample covatiates to be taken into account for training and prediction, e.g. `["age", "cell_type"]`.
            sample_covariate_reps: A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.uns` storing a dictionary with the representation of the covariates. E.g. `{"cell_type": "cell_type_embeddings"}` with `adata.uns["cell_type_embeddings"] = {"cell_typeA": np.array, "cell_typeB": np.array}`.
            split_covariates: Covariates in adata.obs to split all control cells into different control populations. The perturbed cells are also split according to these columns, but if these covariates should also be encoded in the model, the corresponding column should also be used in `perturbation_covariates` or `sample_covariates`.
            max_combination_length: Maximum number of combinations of primary `perturbation_covariates`. If `None`, the value is inferred from the provided `perturbation_covariates`.
            null_value: Value to use for padding to `max_combination_length`.

        Returns
        -------
            None

        """
        self.sample_rep = sample_rep
        self.control_key = control_key
        self.perturbation_covariates = perturbation_covariates
        self.perturbation_covariate_reps = perturbation_covariate_reps
        self.sample_covariates = sample_covariates
        self.sample_covariate_reps = sample_covariate_reps
        self.split_covariates = split_covariates
        self.max_combination_length = max_combination_length
        self.null_value = null_value

        self.pdata = TrainingData.load_from_adata(
            self.adata,
            sample_rep=self.sample_rep,
            control_key=self.control_key,
            perturbation_covariates=self.perturbation_covariates,
            perturbation_covariate_reps=self.perturbation_covariate_reps,
            sample_covariates=self.sample_covariates,
            sample_covariate_reps=self.sample_covariate_reps,
            split_covariates=self.split_covariates,
            max_combination_length=self.max_combination_length,
            null_value=self.null_value,
        )

        self._data_dim = self.pdata.cell_data.shape[-1]

    def prepare_validation_data(
        self,
        adata: ad.AnnData,
        name: str = "validation",
    ) -> None:
        """Prepare validation data.

        Args:
            adata: Anndata object.
            name: Name of the validation data.
            **kwargs: Keyword arguments.

        Returns
        -------
            None
        """
        if self.pdata is None:
            raise ValueError(
                "Dataloader not initialized. Training data needs to be set up before preparing validation data. Please call prepare_data first."
            )
        val_data = ValidationData.load_from_adata(
            adata,
            sample_rep=self.sample_rep,
            control_key=self.control_key,
            perturbation_covariates=self.perturbation_covariates,
            perturbation_covariate_reps=self.perturbation_covariate_reps,
            sample_covariates=self.sample_covariates,
            sample_covariate_reps=self.sample_covariate_reps,
            split_covariates=self.split_covariates,
            max_combination_length=self.max_combination_length,
            null_value=self.null_value,
        )
        self._validation_data[name] = val_data

    def prepare_model(
        self,
        encode_conditions: bool = True,
        condition_embedding_dim: int = 32,
        time_encoder_dims: Sequence[int] = (1024, 1024, 1024),
        time_encoder_dropout: float = 0.0,
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
        hidden_dropout: float = 0.0,
        decoder_dims: Sequence[int] = (1024, 1024, 1024),
        decoder_dropout: float = 0.0,
        condition_encoder_kwargs: dict[str, Any] | None = None,
        velocity_field_kwargs: dict[str, Any] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        flow: (
            dict[Literal["constant_noise", "schroedinger_bridge"], float] | None
        ) = None,
        match_fn: Callable[
            [ArrayLike, ArrayLike], ArrayLike
        ] = solver_utils.match_linear,
        optimizer: optax.GradientTransformation = optax.adam(1e-4),
        seed=0,
    ) -> None:
        """Prepare model for training.

        Args:
            encode_conditions: Whether to encode conditions.
            condition_embedding_dim: Dimensions of the condition embedding.
            condition_encoder_kwargs: Keyword arguments for the condition encoder.
            time_encoder_dims: Dimensions of the time embedding.
            time_encoder_dropout: Dropout rate for the time embedding.
            hidden_dims: Dimensions of the hidden layers.
            hidden_dropout: Dropout rate for the hidden layers.
            decoder_dims: Dimensions of the output layers.
            condition_encoder_kwargs: Keyword arguments for the condition encoder.
            velocity_field_kwargs: Keyword arguments for the velocity field.
            solver_kwargs: Keyword arguments for the solver.
            decoder_dropout: Dropout rate for the output layers.
            flow: Flow to use for training. Shoudl be a dict with the form {"constant_noise": noise_val} or {"schroedinger_bridge": noise_val}.
            match_fn: Matching function.
            optimizer: Optimizer for training.
            seed: Random seed.

        Returns
        -------
            None
        """
        if self.pdata is None:
            raise ValueError(
                "Dataloader not initialized. Please call prepare_data first."
            )

        condition_encoder_kwargs = condition_encoder_kwargs or {}
        velocity_field_kwargs = velocity_field_kwargs or {}
        solver_kwargs = solver_kwargs or {}
        flow = flow or {"constant_noise": 0.0}

        vf = ConditionalVelocityField(
            output_dim=self._data_dim,
            max_combination_length=self.pdata.max_combination_length,
            encode_conditions=encode_conditions,
            condition_embedding_dim=condition_embedding_dim,
            condition_encoder_kwargs=condition_encoder_kwargs,
            time_encoder_dims=time_encoder_dims,
            time_encoder_dropout=time_encoder_dropout,
            hidden_dims=hidden_dims,
            hidden_dropout=hidden_dropout,
            decoder_dims=decoder_dims,
            decoder_dropout=decoder_dropout,
            **velocity_field_kwargs,
        )

        flow, noise = next(iter(flow.items()))
        if flow == "constant_noise":
            flow = dynamics.ConstantNoiseFlow(noise)
        elif flow == "bridge":
            flow = dynamics.BrownianBridge(noise)
        else:
            raise NotImplementedError(
                f"The key of `flow` must be `constant_noise` or `bridge` but found {flow.keys()[0]}."
            )
        if self.solver == "otfm":
            self._solver = otfm.OTFlowMatching(
                vf=vf,
                match_fn=match_fn,
                flow=flow,
                optimizer=optimizer,
                conditions=self.pdata.condition_data,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        elif self.solver == "genot":
            self._solver = genot.GENOT(
                vf=vf,
                data_match_fn=match_fn,
                flow=flow,
                source_dim=self._data_dim,
                target_dim=self._data_dim,
                optimizer=optimizer,
                conditions=self.pdata.condition_data,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        self.trainer = CellFlowTrainer(model=self._solver)

    def train(
        self,
        num_iterations: int,
        batch_size: int = 64,
        valid_freq: int = 10,
        callbacks: Sequence[Callable] = [],
        monitor_metrics: Sequence[str] = [],
    ) -> None:
        """Train the model.

        Args:
            num_iterations: Number of iterations to train the model.
            batch_size: Batch size.
            valid_freq: Frequency of validation.
            callbacks: Callback functions.
            monitor_metrics: Metrics to monitor.

        Returns
        -------
            None
        """
        if self.pdata is None:
            raise ValueError("Data not initialized. Please call prepare_data first.")

        if self.trainer is None:
            raise ValueError("Model not initialized. Please call prepare_model first.")

        self.dataloader = TrainSampler(data=self.pdata, batch_size=batch_size)

        self.trainer.train(
            dataloader=self.dataloader,
            num_iterations=num_iterations,
            valid_freq=valid_freq,
            valid_data=self._validation_data,
            callbacks=callbacks,
            monitor_metrics=monitor_metrics,
        )

    def predict(
        self,
        adata: ad.AnnData,
    ) -> ArrayLike:
        """Predict perturbation.

        Args:
            adata: Anndata object.


        Returns
        -------
            Perturbation prediction.
        """
        pass

    def get_condition_embedding(
        self,
        adata: ad.AnnData,
    ) -> ArrayLike:
        """Get condition embedding.

        Args:
            adata: Anndata object.

        Returns
        -------
            Condition embedding.
        """
        pass

    def save(
        self,
        dir_path: str,
        file_prefix: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save the model. Pickles the CellFlow class instance.

        Args:
        dir_path: Path to a directory, defaults to current directory
        file_prefix: Prefix to prepend to the file name.
        overwrite: Overwrite existing data or not.

        Returns
        -------
        None
        """
        file_name = (
            f"{file_prefix}_{self.__class__.__name__}.pkl"
            if file_prefix is not None
            else f"{self.__class__.__name__}.pkl"
        )
        file_dir = (
            os.path.join(dir_path, file_name) if dir_path is not None else file_name
        )

        if not overwrite and os.path.exists(file_dir):
            raise RuntimeError(
                f"Unable to save to an existing file `{file_dir}` use `overwrite=True` to overwrite it."
            )
        with open(file_dir, "wb") as f:
            cloudpickle.dump(self, f)
