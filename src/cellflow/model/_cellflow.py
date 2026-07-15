"""In-memory (legacy) CellFlow: training data materialized from an ``AnnData`` via :meth:`prepare_data`.

The streaming path (:class:`~cellflow.model.CellFlowAnnbatch`) is the default; this in-memory path is
kept for backward compatibility.
"""

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import anndata as ad
import numpy as np

from cellflow.data._datamanager import DataManager
from cellflow.data._legacy import OOCTrainSampler, TrainingData, TrainSampler, ValidationSampler
from cellflow.model._base import BaseCellFlow

__all__ = ["CellFlow"]


class CellFlow(BaseCellFlow):
    """CellFlow with in-memory training data extracted from an :class:`~anndata.AnnData`.

    Prepare training data with :meth:`prepare_data`; model setup, training, prediction and IO are
    inherited from :class:`~cellflow.model._base.BaseCellFlow`. For large or out-of-core datasets use
    :class:`~cellflow.model.CellFlowAnnbatch` (the default streaming path).
    """

    def __init__(self, adata: ad.AnnData | None = None, solver: Literal["otfm", "genot"] = "otfm"):
        super().__init__(solver)
        if adata is not None:
            warnings.warn(
                "Passing `adata` to `CellFlow(...)` is deprecated and will be removed in a future "
                "release; pass it to `prepare_data(adata=...)` instead.",
                FutureWarning,
                stacklevel=2,
            )
        self._adata = adata
        self._train_data: TrainingData | None = None

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
        adata: ad.AnnData | None = None,
    ) -> None:
        """Prepare the dataloader for training from :attr:`~cellflow.model.CellFlow.adata`.

        Parameters
        ----------
        sample_rep
            Key in :attr:`~anndata.AnnData.obsm` of :attr:`cellflow.model.CellFlow.adata` where
            the sample representation is stored or ``'X'`` to use :attr:`~anndata.AnnData.X`.
        control_key
            Key of a boolean column in :attr:`~anndata.AnnData.obs` of
            :attr:`cellflow.model.CellFlow.adata` that defines the control samples.
        perturbation_covariates
            A dictionary where the keys indicate the name of the covariate group and the values are
            keys in :attr:`~anndata.AnnData.obs` of :attr:`cellflow.model.CellFlow.adata`. The
            corresponding columns can be of the following types:

            - categorial: The column contains categories whose representation is stored in
              :attr:`~anndata.AnnData.uns`, see ``'perturbation_covariate_reps'``.
            - boolean: The perturbation is present or absent.
            - numeric: The perturbation is given as a numeric value, possibly linked to
              a categorical perturbation, e.g. dosages for a drug.

            If multiple groups are provided, the first is interpreted as the primary
            perturbation and the others as covariates corresponding to these perturbations.
        perturbation_covariate_reps
            A :class:`dict` where the keys indicate the name of the covariate group and the values
            are keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the representation
            of the covariates.
        sample_covariates
            Keys in :attr:`~anndata.AnnData.obs` indicating sample covariates. Sample covariates
            are defined such that each cell has only one value for each sample covariate (in
            constrast to ``'perturbation_covariates'`` which can have multiple values for each
            cell). If :obj:`None`, no sample
        sample_covariate_reps
            A dictionary where the keys indicate the name of the covariate group and the values
            are keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the representation
            of the covariates.
        split_covariates
            Covariates in :attr:`~anndata.AnnData.obs` to split all control cells into different
            control populations. The perturbed cells are also split according to these columns,
            but if any of the ``'split_covariates'`` has a representation which should be
            incorporated by the model, the corresponding column should also be used in
            ``'perturbation_covariates'``.
        max_combination_length
            Maximum number of combinations of primary ``'perturbation_covariates'``. If
            :obj:`None`, the value is inferred from the provided ``'perturbation_covariates'``
            as the maximal number of perturbations a cell has been treated with.
        null_value
            Value to use for padding to ``'max_combination_length'``.
        adata
            The :class:`~anndata.AnnData` object to extract the training data from. If :obj:`None`,
            the object passed to the (deprecated) constructor argument is used instead.

        Returns
        -------
        Updates the following fields:

        - :attr:`cellflow.model.CellFlow.data_manager` - the :class:`cellflow.data.DataManager` object.
        - :attr:`cellflow.model.CellFlow.train_data` - the training data.

        Example
        -------
            Consider the case where we have combinations of drugs along with dosages, saved in
            :attr:`~anndata.AnnData.obs` as columns ``drug_1`` and ``drug_2`` with three different
            drugs ``DrugA``, ``DrugB``, and ``DrugC``, and ``dose_1`` and ``dose_2`` for their
            dosages, respectively. We store the embeddings of the drugs in
            :attr:`~anndata.AnnData.uns` under the key ``drug_embeddings``, while the dosage
            columns are numeric. Moreover, we have a covariate ``cell_type`` with values
            ``cell_typeA`` and ``cell_typeB``, with embeddings stored in
            :attr:`~anndata.AnnData.uns` under the key ``cell_type_embeddings``. Note that we then
            also have to set ``'split_covariates'`` as we assume we have an unperturbed population
            for each cell type.

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
        adata = adata if adata is not None else self._adata
        if adata is None:
            raise ValueError(
                "No `adata` provided. Pass it as `prepare_data(adata=...)` (recommended) or "
                "construct `CellFlow(adata=...)`."
            )
        self._adata = adata  # kept for downstream predict / validation / plotting

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

        self.train_data = self._dm.get_train_data(self.adata)
        self._data_dim = self.train_data.cell_data.shape[-1]  # type: ignore[union-attr]

    def prepare_validation_data(
        self,
        adata: ad.AnnData,
        name: str,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Prepare the validation data.

        Validation is always in-memory (metrics need materialized cells): pass a held-out
        :class:`~anndata.AnnData` and its cells (at ``sample_rep``) + condition embeddings become a
        ``ValidationData``. Works for both the in-memory and streaming training paths; in the streaming
        path the ``adata`` must carry the same ``sample_rep`` and the covariate embeddings in ``.uns``.
        (Unrelated to the ``val`` split from :meth:`split_annbatch_data`, a streaming loader — not a
        validation set.)

        Parameters
        ----------
        adata
            An :class:`~anndata.AnnData` object.
        name
            Name of the validation data defining the key in
            :attr:`cellflow.model.CellFlow.validation_data`.
        n_conditions_on_log_iteration
            Number of conditions to use for computation callbacks at each logged iteration.
            If :obj:`None`, use all conditions.
        n_conditions_on_train_end
            Number of conditions to use for computation callbacks at the end of training.
            If :obj:`None`, use all conditions.
        predict_kwargs
            Keyword arguments for the prediction function
            :func:`cellflow.solvers._otfm.OTFlowMatching.predict` or
            :func:`cellflow.solvers._genot.GENOT.predict` used during validation.

        Returns
        -------
        :obj:`None`, and updates the following fields:

        - :attr:`cellflow.model.CellFlow.validation_data` - a dictionary with the validation data.

        """
        if self.train_data is None:
            raise ValueError("Model data not initialized. Call `prepare_data(...)` before preparing validation data.")
        val_data = self._dm.get_validation_data(
            adata,
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        self._validation_data[name] = val_data
        predict_kwargs = predict_kwargs or {}
        if (
            "predict_kwargs" in self._validation_data
            and len(self._validation_data["predict_kwargs"]) > 0
            and len(predict_kwargs) > 0
        ):
            self._validation_data["predict_kwargs"].update(predict_kwargs)
            predict_kwargs = self._validation_data["predict_kwargs"]
        self._validation_data["predict_kwargs"] = predict_kwargs

    @property
    def train_data(self) -> TrainingData | None:
        """The training data."""
        return self._train_data

    @train_data.setter
    def train_data(self, data: TrainingData) -> None:
        """Set the training data."""
        if not isinstance(data, TrainingData):
            raise ValueError(f"Expected `data` to be an instance of `TrainingData`, found `{type(data)}`.")
        self._train_data = data

    # ── path-specific hook implementations ───────────────────────────────────────────────────────
    def _encoder_conditions(self) -> tuple[dict[str, np.ndarray], int]:
        if self.train_data is None:
            raise ValueError("Data not initialized. Please call `prepare_data(...)` first.")
        return self.train_data.condition_data, self.train_data.max_combination_length

    def _bind_train_dataloader(self, batch_size: int, out_of_core_dataloading: bool) -> None:
        self._dataloader = (
            OOCTrainSampler(data=self.train_data, batch_size=batch_size)
            if out_of_core_dataloading
            else TrainSampler(data=self.train_data, batch_size=batch_size)
        )

    def _build_validation_loaders(self) -> dict[str, Any]:
        return {k: ValidationSampler(v) for k, v in self.validation_data.items() if k != "predict_kwargs"}
