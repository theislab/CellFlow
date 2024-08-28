import functools
import os
import types
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import anndata as ad
import cloudpickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from numpy.typing import ArrayLike
from ott.neural.methods.flows import dynamics

from cfp import _constants
from cfp._types import Layers_separate_input_t, Layers_t
from cfp.data._data import ConditionData, ValidationData
from cfp.data._dataloader import PredictionSampler, TrainSampler, ValidationSampler
from cfp.data._datamanager import DataManager
from cfp.model._utils import _write_predictions
from cfp.networks._velocity_field import ConditionalVelocityField
from cfp.plotting import _utils
from cfp.solvers import _genot, _otfm
from cfp.training._callbacks import BaseCallback
from cfp.training._trainer import CellFlowTrainer
from cfp.utils import match_linear

__all__ = ["CellFlow"]


class CellFlow:
    """CellFlow model for perturbation prediction using Flow Matching and Optimal Transport.

    CellFlow builds upon neural optimal transport estimators extending :cite:`tong:23`,
    :cite:`pooladian:23`, :cite:`eyring:24`, :cite:`klein:23` which are all based on
    Flow Matching :cite:`lipman:22`.

    Parameters
    ----------
        adata
            An :class:`anndata.AnnData` object to extract the training data from.
        solver
            Solver to use for training. Either ``'otfm'`` or ``'genot'``.
    """

    def __init__(self, adata: ad.AnnData, solver: Literal["otfm", "genot"] = "otfm"):

        self._adata = adata
        self._solver_class = _otfm.OTFlowMatching if solver == "otfm" else _genot.GENOT
        self._dataloader: TrainSampler | None = None
        self._trainer: CellFlowTrainer | None = None
        self._validation_data: dict[str, ValidationData] = {}
        self._solver: _otfm.OTFlowMatching | _genot.GENOT | None = None
        self._condition_dim: int | None = None
        self._vf: ConditionalVelocityField | None = None

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
        """Prepare the dataloader for training from :attr:`cfp.model.CellFlow.adata`.

        Parameters
        ----------
        sample_rep
            Key in :attr:`~anndata.AnnData.obsm` of :attr:`cfp.model.CellFlow.adata` where the sample representation is stored or ``'X'`` to use
            :attr:`~anndata.AnnData.X`.
        control_key
            Key of a boolean column in :attr:`~anndata.AnnData.obs` of :attr:`cfp.model.CellFlow.adata` that defines the control samples.
        perturbation_covariates
            A dictionary where the keys indicate the name of the covariate group and the values are
            keys in :attr:`~anndata.AnnData.obs` of :attr:`cfp.model.CellFlow.adata`. The corresponding columns can be of the following types:

            - categorial: The column contains categories whose representation is stored in
              :attr:`~anndata.AnnData.uns`, see ``'perturbation_covariate_reps'``.
            - boolean: The perturbation is present or absent.
            - numeric: The perturbation is given as a numeric value, possibly linked to a categorical
              perturbation, e.g. dosages for a drug.

            If multiple groups are provided, the first is interpreted as the primary
            perturbation and the others as covariates corresponding to these perturbations.
        perturbation_covariate_reps
            A :class:`dict` where the keys indicate the name of the covariate group and the values are
            keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the representation of
            the covariates.
        sample_covariates
            Keys in :attr:`~anndata.AnnData.obs` indicating sample covariates. Sample covariates are
            defined such that each cell has only one value for each sample covariate (in constrast to
            ``'perturbation_covariates'`` which can have multiple values for each cell). If :obj:`None`, no sample
        sample_covariate_reps
            A dictionary where the keys indicate the name of the covariate group and the values
            are keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the representation
            of the covariates.
        split_covariates
            Covariates in :attr:`~anndata.AnnData.obs` to split all control cells into different
            control populations. The perturbed cells are also split according to these columns,
            but if any of the ``'split_covariates'`` has a representation which should be incorporated by
            the model, the corresponding column should also be used in ``'perturbation_covariates'``.
        max_combination_length
            Maximum number of combinations of primary ``'perturbation_covariates'``. If :obj:`None`, the
            value is inferred from the provided ``'perturbation_covariates'`` as the maximal number of
            perturbations a cell has been treated with.
        null_value
            Value to use for padding to ``'max_combination_length'``.

        Returns
        -------
        Updates the following fields:

        - :attr:`cfp.model.CellFlow.dm` - the :class:`cfp.data.DataManager` object.
        - :attr:`cfp.model.CellFlow.train_data` - the training data.

        Example
        -------
            Consider the case where we have combinations of drugs along with dosages, saved in :attr:`anndata.AnnData.obs` as
            columns `drug_1` and `drug_2` with three different drugs `DrugA`, `DrugB`, and `DrugC`, and `dose_1` and `dose_2`
            for their dosages, respectively. We store the embeddings of the drugs in
            :attr:`anndata.AnnData.uns` under the key `drug_embeddings`, while the dosage columns are numeric. Moreover,
            we have a covariate `cell_type` with values `cell_typeA` and `cell_typeB`, with embeddings stored in
            :attr:`anndata.AnnData.uns` under the key `cell_type_embeddings`. Note that we then also have to set ``'split_covariates'``
            as we assume we have an unperturbed population for each cell type.

            .. code-block:: python

                perturbation_covariates = {{"drug": ("drug_1", "drug_2"), "dose": ("dose_1", "dose_2")}}
                perturbation_covariate_reps = {"drug": "drug_embeddings"}
                adata.uns["drug_embeddings"] = {
                    "drugA": np.array([0.1, 0.2, 0.3]),
                    "drugB": np.array([0.4, 0.5, 0.6]),
                    "drugC": np.array([-0.2, 0.3, 0.0]),
                }

                sample_covariates = {"cell_type": "cell_type_embeddings"}
                adata.uns["cell_type_embeddings"] = {
                    "cell_typeA": np.array([0.0, 1.0]),
                    "cell_typeB": np.array([0.0, 2.0]),
                }

                split_covariates = ["cell_type"]

                cf = CellFlow(adata)
                cf = cf.prepare_data(
                    sample_rep="X",
                    control_key="control",
                    perturbation_covariates=perturbation_covariates,
                    perturbation_covariate_reps=perturbation_covariate_reps,
                    sample_covariates=sample_covariates,
                    sample_covariate_reps=sample_covariate_reps,
                    split_covariates=split_covariates,
                )
        """
        self._dm = DataManager(
            self.adata,
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
            split_covariates=split_covariates,
            max_combination_length=max_combination_length,
            null_value=null_value,
        )

        # TODO: rename to self.train_data
        self.train_data = self._dm.get_train_data(self.adata)

        self._data_dim = self.train_data.cell_data.shape[-1]

    def prepare_validation_data(
        self,
        adata: ad.AnnData,
        name: str,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
    ) -> None:
        """Prepare the validation data.

        Parameters
        ----------
        adata
            An :class:`anndata.AnnData` object.
        name
            Name of the validation data defining the key in :attr:`validation_data`.
        n_conditions_on_log_iterations
            Number of conditions to use for computation callbacks at each logged iteration.
            If :obj:`None`, use all conditions.
        n_conditions_on_train_end
            Number of conditions to use for computation callbacks at the end of training.
            If :obj:`None`, use all conditions.

        Returns
        -------
        :obj:`None`, and updates the following fields:

        - :attr:`cfp.model.CellFlow.validation_data` - a dictionary with the validation data.

        """
        if self.train_data is None:
            raise ValueError(
                "Dataloader not initialized. Training data needs to be set up before preparing validation data. Please call prepare_data first."
            )
        val_data = self._dm.get_validation_data(
            adata,
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        self._validation_data[name] = val_data

    def prepare_model(
        self,
        encode_conditions: bool = True,
        condition_embedding_dim: int = 32,
        pooling: Literal[
            "mean", "attention_token", "attention_seed"
        ] = "attention_token",
        pooling_kwargs: dict[str, Any] = types.MappingProxyType({}),
        time_encoder_dims: Sequence[int] = (1024, 1024, 1024),
        time_encoder_dropout: float = 0.0,
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
        hidden_dropout: float = 0.0,
        decoder_dims: Sequence[int] = (1024, 1024, 1024),
        decoder_dropout: float = 0.0,
        layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(
            default_factory=lambda: []
        ),
        layers_after_pool: Layers_t = dc_field(default_factory=lambda: []),
        cond_output_dropout: float = 0.0,
        condition_encoder_kwargs: dict[str, Any] | None = None,
        pool_sample_covariates: bool = True,
        vf_act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu,
        time_freqs: int = 1024,
        solver_kwargs: dict[str, Any] | None = None,
        flow: dict[Literal["constant_noise", "bridge"], float] | None = None,
        match_fn: Callable[[ArrayLike, ArrayLike], ArrayLike] = match_linear,
        optimizer: optax.GradientTransformation = optax.adam(1e-4),
        seed=0,
    ) -> None:
        """Prepare the model for training.

        This function sets up the neural network architecture and specificities of the :attr:`solver`.
        When :attr:`solver` is an instance of :class:`cfp.solvers._genot.GENOT`, the following arguments have
        to be passed to ``'condition_encoder_kwargs'``:

        - ``'genot_source_layers'``: Layers processing the cell data serving as an additional condition to the model (see :cite:`klein:23`).
        - ``'genot_source_dim'``: Dimension of the source data.


        Parameters
        ----------
        encode_conditions
            Processes the embedding of the perturbation conditions if :obj:`True`. If :obj:`False`,
            directly inputs the embedding of the perturbation conditions to the generative velocity field.
            In the latter case, ``'condition_embedding_dim'``, ``'condition_encoder_kwargs'``, ``'pooling'``,
            ``'pooling_kwargs'``, ``'layers_before_pool'``, ``'layers_after_pool'``, ``'cond_output_dropout'``
            are ignored.
        pooling
            Pooling method, should be one of:

            - ``'mean'``: Aggregates combinations of covariates by the mean of their learned embeddings.
            - ``'attention_token'``: Aggregates combinations of covariates by an attention mechanism with a class
              token.
            - ``'attention_seed'``: Aggregates combinations of covariates by seed attention.

        pooling_kwargs
            Keyword arguments for the pooling method corresponding to:

            - :class:`cfp.networks.TokenAttentionPooling` if ``'pooling'`` is ``'attention_token'``.
            - :class:`cfp.networks.SeedAttentionPooling` if ``'pooling'`` is ``'attention_seed'``.

        condition_embedding_dim
            Dimensions of the condition embedding, i.e. the last layer of the
            :class:`cfp.networks.ConditionEncoder`.
        condition_encoder_kwargs
            Keyword arguments for the :class:`cfp.networks.ConditionEncoder`.
        time_encoder_dims
            Dimensions of the layers processing the time embedding in
            :attr:`cfp.networks.ConditionalVelocityField.time_encoder`.
        time_encoder_dropout
            Dropout rate for the :attr:`cfp.networks.ConditionalVelocityField.time_encoder`.
        hidden_dims
            Dimensions of the layers processing the input to the velocity field
            via :attr:`cfp.networks.ConditionalVelocityField.x_encoder`.
        hidden_dropout
            Dropout rate for :attr:`cfp.networks.ConditionalVelocityField.x_encoder`.
        decoder_dims
            Dimensions of the output layers in :attr:`cfp.networks.ConditionalVelocityField.decoder`.
        layers_before_pool
            Layers applied to the condition embeddings before pooling. Can be of type

            - :class:`Tuple` with elements corresponding to dictionaries with keys:

                - ``'layer_type'`` of type :class:`str` indicating the type of the layer, can be
                  ``'mlp'`` or ``'self_attention'``.
                - Further keyword arguments for the layer type :class:`cfp.networks.MLPBlock` or
                  :class:`cfp.networks.SelfAttentionBlock`.

            - :class:`dict` with keys corresponding to perturbation covariate keys, and values correspondinng to
              the above mentioned tuples.

        layers_after_pool
            Layers applied to the condition embeddings after pooling, and before applying the last layer of
            size ``'condition_embedding_dim'``. Should be of type :class:`Tuple` with elements corresponding to
            dictionaries with keys:

            - ``'layer_type'`` of type :class:`str` indicating the type of the layer, can be
              ``'mlp'`` or ``'self_attention'``.
            - Further keys depend on the layer type, either for :class:`cfp.networks.MLPBlock` or for
              :class:`cfp.networks.SelfAttentionBlock`.

        cond_output_dropout
            Dropout rate for the last layer of the :class:`cfp.networks.ConditionEncoder`.
        condition_encoder_kwargs
            Keyword arguments for the :class:`cfp.networks.ConditionEncoder`.
        pool_sample_covariates
            Whether to include sample covariates in the pooling.
        vf_act_fn
            Activation function of the :class:`cfp.networks.ConditionalVelocityField`.
        time_freqs
            Frequency of the time encoding (:func:`ott.neural.networks.layers.cyclical_time_encoder`).
        solver_kwargs
            Keyword arguments for the solver :class:`cfp.solvers.OTFlowMatching` or :class:`cfp.solvers.GENOT`.
        decoder_dropout
            Dropout rate for the output layer :attr:`cfp.networks.ConditionalVelocityField.decoder`.
        flow
            Flow to use for training. Should be a :class:`dict` of the form

            - ``'{"constant_noise": noise_val'``
            - ``'{"bridge": noise_val}'``

            If :obj:`None`, defaults to ``'{"constant_noise": 0.0}'``.
        match_fn
            Matching function between unperturbed and perturbed cells. Should take as input source and target
            data and return the optimal transport matrix, see e.g. :func:`cfp.utils.match_linear`.
        optimizer
            Optimizer used for training.
        seed
            Random seed.

        Returns
        -------
        Updates the following fields:

        - :attr:`cfp.model.CellFlow.vf` - an instance of the :class:`cfp.networks.ConditionalVelocityField`.
        - :attr:`cfp.model.CellFlow.solver` - an instance of :class:`cfp.solvers.OTFlowMatching` or :class:`cfp.solvers.GENOT`.
        - :attr:`cfp.model.CellFlow.trainer` - an instance of the :class:`cfp.training.CellFlowTrainer`.
        """
        flow = flow or {"constant_noise": 0.0}
        if self.train_data is None:
            raise ValueError(
                "Dataloader not initialized. Please call `prepare_data` first."
            )

        condition_encoder_kwargs = condition_encoder_kwargs or {}
        covariates_not_pooled = (
            [] if pool_sample_covariates else self._dm.sample_covariates
        )
        solver_kwargs = solver_kwargs or {}
        flow = flow or {"constant_noise": 0.0}

        self.vf = ConditionalVelocityField(
            output_dim=self._data_dim,
            max_combination_length=self.train_data.max_combination_length,
            encode_conditions=encode_conditions,
            condition_embedding_dim=condition_embedding_dim,
            covariates_not_pooled=covariates_not_pooled,
            pooling=pooling,
            pooling_kwargs=pooling_kwargs,
            layers_before_pool=layers_before_pool,
            layers_after_pool=layers_after_pool,
            cond_output_dropout=cond_output_dropout,
            condition_encoder_kwargs=condition_encoder_kwargs,
            act_fn=vf_act_fn,
            time_freqs=time_freqs,
            time_encoder_dims=time_encoder_dims,
            time_encoder_dropout=time_encoder_dropout,
            hidden_dims=hidden_dims,
            hidden_dropout=hidden_dropout,
            decoder_dims=decoder_dims,
            decoder_dropout=decoder_dropout,
        )

        flow, noise = next(iter(flow.items()))
        if flow == "constant_noise":
            flow = dynamics.ConstantNoiseFlow(noise)
        elif flow == "bridge":
            flow = dynamics.BrownianBridge(noise)
        else:
            raise NotImplementedError(
                f"The key of `flow` must be `'constant_noise'` or `'bridge'` but found {flow}."
            )

        if self._solver_class == _otfm.OTFlowMatching:
            self._solver = self._solver_class(
                vf=self.vf,
                match_fn=match_fn,
                flow=flow,
                optimizer=optimizer,
                conditions=self.train_data.condition_data,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        elif self._solver_class == _genot.GENOT:
            self._solver = self._solver_class(
                vf=self.vf,
                data_match_fn=match_fn,
                flow=flow,
                source_dim=self._data_dim,
                target_dim=self._data_dim,
                optimizer=optimizer,
                conditions=self.train_data.condition_data,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Solver must be an instance of OTFlowMatching or GENOT, got {type(self.solver)}"
            )
        self._trainer = CellFlowTrainer(solver=self.solver)  # type: ignore[arg-type]

    def train(
        self,
        num_iterations: int,
        batch_size: int = 1024,
        valid_freq: int = 1000,
        callbacks: Sequence[BaseCallback] = [],
        monitor_metrics: Sequence[str] = [],
    ) -> None:
        """Train the model.

        Note
        ----
        A low value of ``'valid_freq'`` results in long training
        because predictions are time-consuming compared to training steps.

        Parameters
        ----------
        num_iterations
            Number of iterations to train the model.
        batch_size
            Batch size.
        valid_freq
            Frequency of validation.
        callbacks
            Callbacks to perform at each validation step. There are two types of callbacks:
            - Callbacks for computations should inherit from :class:`cfp.training.ComputationCallback:` see e.g.
              :class:`cfp.training.Metrics`.
            - Callbacks for logging should inherit from :class:`~cfp.training.LoggingCallback` see e.g.
              :class:`~cfp.training.WandbLogger`.
        monitor_metrics
            Metrics to monitor.

        Returns
        -------
        Updates the following fields:

        - :attr:`cfp.model.CellFlow.dataloader` - the training dataloader.
        - :attr:`cfp.model.CellFlow.solver` - the trained solver.
        """
        if self.train_data is None:
            raise ValueError("Data not initialized. Please call `prepare_data` first.")

        if self.trainer is None:
            raise ValueError(
                "Model not initialized. Please call `prepare_model` first."
            )

        self._dataloader = TrainSampler(data=self.train_data, batch_size=batch_size)
        validation_loaders = {
            k: ValidationSampler(v) for k, v in self.validation_data.items()
        }

        self._solver = self.trainer.train(
            dataloader=self._dataloader,
            num_iterations=num_iterations,
            valid_freq=valid_freq,
            valid_loaders=validation_loaders,
            callbacks=callbacks,
            monitor_metrics=monitor_metrics,
        )

    def predict(
        self,
        adata: ad.AnnData,
        sample_rep: str,
        covariate_data: pd.DataFrame,
        condition_id_key: str | None = None,
        key_added_prefix: str | None = _constants.PREDICTION_PREFIX,
        n_samples: int = 1,
        **kwargs: Any,
    ) -> dict[str, dict[str, ArrayLike]] | dict[str, ArrayLike]:
        """Predict perturbation responses.

        Parameters
        ----------
        adata
            An :class:`~anndata.AnnData` object with the source representation.
        sample_rep
            Key in :attr:`~anndata.AnnData.obsm` where the sample representation is stored or
            ``'X'`` to use :attr:`~anndata.AnnData.X`.
        covariate_data
            Covariate data defining the condition to predict. This :class:`~pandas.DataFrame` should have the same
            columns as :attr:`~anndata.AnnData.obs` of :attr:`cfp.model.CellFlow.adata`, and
            as registered in :attr:`cfp.model.CellFlow.dm`.
        condition_id_key
            Key in ``'covariate_data'`` defining the condition name.
        key_added_prefix
            Prefix to store the prediction in :attr:`~anndata.AnnData.obsm`.
        n_samples
            Number of perturbed cells to generate for each single cell in the control population.
            Only available if :attr:`cfp.model.CellFlow.solver` is an instance of :class:`cfp.solvers.GENOT`.
        kwargs
            Keyword arguments for the predict function, i.e. :meth:`cfp.solvers.OTFlowMatching.predict` or
            :meth:`cfp.solvers.GENOT.predict`.

        Returns
        -------
        A :class:`dict` with the predicted sample representation for each perturbation.
        """
        if not self.solver.is_trained:  # type: ignore[union-attr]
            raise ValueError("Model not trained. Please call `train` first.")

        if n_samples > 1:
            if not isinstance(self.solver, _genot.GENOT):
                raise ValueError(
                    "Multiple samples can only be generated if the solver is `genot`."
                )
            kwargs["n_samples"] = n_samples
        if adata is not None and covariate_data is not None:
            if self._dm.control_key not in adata.obs.columns:
                raise ValueError(
                    f"If both `adata` and `covariate_data` are given, the control key `{self._dm.control_key}` must be in `adata.obs`."
                )
            if not adata.obs[self._dm.control_key].all():
                raise ValueError(
                    f"If both `adata` and `covariate_data` are given, all samples in `adata` must be control samples, and thus `adata.obs[`{self._dm.control_key}`] must be set to `True` everywhere."
                )
        pred_data = self._dm.get_prediction_data(
            adata,
            sample_rep=sample_rep,
            covariate_data=covariate_data,
            condition_id_key=condition_id_key,
        )
        pred_loader = PredictionSampler(pred_data)
        batch = pred_loader.sample()
        src = batch["source"]
        condition = batch.get("condition", None)
        out = jax.tree.map(functools.partial(self.solver.predict, **kwargs), src, condition)  # type: ignore[union-attr]
        if key_added_prefix is not None:
            if len(pred_data.control_to_perturbation) > 1:
                raise ValueError(
                    f"When saving predictions to `adata`, all control cells must be from the same control \
                                  population, but found {len(pred_data.control_to_perturbation)} control populations."
                )
            _write_predictions(
                adata=adata,
                predictions=out,
                key_added_prefix=key_added_prefix,
            )
        return out

    def get_condition_embedding(
        self,
        covariate_data: pd.DataFrame | ConditionData,
        rep_dict: dict[str, str] | None = None,
        condition_id_key: str | None = None,
        key_added: str | None = _constants.CONDITION_EMBEDDING,
    ) -> pd.DataFrame:
        """Get the embedding of the conditions.

        Outputs the embeddings generated by the :class:`cfp.networks.ConditionEncoder`.

        Parameters
        ----------
        covariate_data
            Can be one of

            - a :class:`pandas.DataFrame` defining the conditions with the same columns as the
              :class:`anndata.AnnData` used for the initialisation of :class:`cfp.model.CellFlow`.
            - an instance of :class:`cfp.data.ConditionData`.

        rep_dict
            Dictionary containing the representations of the perturbation covaraiates.
        condition_id_key
            Key defining the name of the condition. Only available
            if ``'covariate_data'`` is a :class:`pandas.DataFrame`.
        key_added
            Key to store the condition embedding in :attr:`~anndata.AnnData.uns`.

        Returns
        -------
        A :class:`pandas.DataFrame` with the condition embeddings.
        """
        if self.solver is None:
            raise ValueError("Model not trained. Please call `train` first.")

        if not self._dm.is_conditional:
            raise ValueError(
                "Model is not conditional. Condition embeddings are not available."
            )

        if hasattr(covariate_data, "condition_data"):
            cond_data = covariate_data
        elif isinstance(covariate_data, pd.DataFrame):
            cond_data = self._dm.get_condition_data(
                covariate_data=covariate_data,
                rep_dict=rep_dict,
                condition_id_key=condition_id_key,
            )
        else:
            raise ValueError(
                "Covariate data must be a `pandas.DataFrame` or an instance of `BaseData`."
            )

        condition_embeddings: dict[str, ArrayLike] = {}
        n_conditions = len(next(iter(cond_data.condition_data.values())))
        for i in range(n_conditions):
            condition = {k: v[[i], :] for k, v in cond_data.condition_data.items()}
            if len(cond_data.perturbation_idx_to_id):
                c_key = cond_data.perturbation_idx_to_id[i]
            else:
                cov_combination = cond_data.perturbation_idx_to_covariates[i]
                c_key = tuple(cov_combination[i] for i in range(len(cov_combination)))
            condition_embeddings[c_key] = self.solver.get_condition_embedding(condition)

        df = pd.DataFrame.from_dict(
            {k: v[0] for k, v in condition_embeddings.items()}  # type: ignore[index]
        ).T
        indices = list(self._dm.sample_covariates)
        for pert_cov in self._dm.perturbation_covariates:
            indices += [
                f"{pert_cov}_{i}" for i in range(self._dm.max_combination_length)
            ]
        df.index.set_names(indices, inplace=True)
        df.drop_duplicates(inplace=True)

        if key_added is not None:
            _utils.set_plotting_vars(self.adata, key=key_added, value=df)

        return df

    def save(
        self,
        dir_path: str,
        file_prefix: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save the model.

        Pickles the CellFlow object.

        Parameters
        ----------
            dir_path
                Path to a directory, defaults to current directory
            file_prefix
                Prefix to prepend to the file name.
            overwrite
                Overwrite existing data or not.

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

    @classmethod
    def load(
        cls,
        filename: str,
    ) -> "CellFlow":
        """
        Load a :class:`cfp.model.CellFlow` model from a saved instance.

        Parameters
        ----------
            filename
                Path to the saved file.

        Returns
        -------
        Loaded instance of the model.
        """
        # Check if filename is a directory
        file_name = (
            os.path.join(filename, f"{cls.__name__}.pkl")
            if os.path.isdir(filename)
            else filename
        )

        with open(file_name, "rb") as f:
            model = cloudpickle.load(f)

        if type(model) is not cls:
            raise TypeError(
                f"Expected the model to be type of `{cls}`, found `{type(model)}`."
            )
        return model

    @property
    def adata(self) -> ad.AnnData:
        """The :class:`~anndata.AnnData` object used for training."""
        return self._adata

    @property
    def solver(self) -> _otfm.OTFlowMatching | _genot.GENOT | None:
        """The solver."""
        return self._solver

    @property
    def dataloader(self) -> TrainSampler | None:
        """The dataloader used for training."""
        return self._dataloader

    @property
    def trainer(self) -> CellFlowTrainer | None:
        """The trainer used for training."""
        return self._trainer

    @property
    def validation_data(self) -> dict[str, ValidationData]:
        """The validation data."""
        return self._validation_data

    @property
    def data_manager(self) -> DataManager:
        """The data manager, initialised with :attr:`cfp.model.CellFlow.adata`."""
        return self._dm

    @property
    def velocity_field(self) -> ConditionalVelocityField | None:
        """The conditional velocity field."""
        return self._vf

    @velocity_field.setter
    def velocity_field(self, vf: ConditionalVelocityField) -> None:
        """Set the velocity field."""
        if not isinstance(vf, ConditionalVelocityField):
            raise ValueError(
                f"Expected `vf` to be an instance of `ConditionalVelocityField`, found `{type(vf)}`."
            )
        self._vf = vf
