from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import anndata
import dask
import dask.dataframe as dd
import dask.delayed
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.preprocessing as preprocessing
from dask.diagnostics import ProgressBar
from pandas.api.types import is_numeric_dtype

from cellflow._logging import logger
from cellflow._types import ArrayLike
from cellflow.data._data import ConditionData, PredictionData, ReturnData, TrainingData, ValidationData

from ._utils import _flatten_list, _to_list

__all__ = ["DataManager"]


class DataManager:
    """Data manager for handling perturbation data.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    covariate_encoder
        Encoder for the primary covariate.
    categorical
        Whether the primary covariate is categorical.
    max_combination_length
        Maximum number of combinations of primary ``perturbation_covariates``.
    sample_rep
        Key in :attr:`~anndata.AnnData.obsm` where the sample representation is stored or ``'X'``
        to use :attr:`~anndata.AnnData.X`.
    covariate_data
        Dataframe with covariates. If :obj:`None`, :attr:`~anndata.AnnData.obs` is used.
    condition_id_key
        Key in :attr:`~anndata.AnnData.obs` that defines the condition id.
    perturbation_covariates
        A dictionary where the keys indicate the name of the covariate group
        and the values are keys in :attr:`~anndata.AnnData.uns`. The corresponding
        columns should be either boolean (presence/abscence of the perturbation) or
        numeric (concentration or magnitude of the perturbation). If multiple groups
        are provided, the first is interpreted as the primary perturbation and the
        others as covariates corresponding to these perturbations, e.g.
        ``{"drug":("drugA", "drugB"), "time":("drugA_time", "drugB_time")}``.
    perturbation_covariate_reps
        A dictionary where the keys indicate the name of the covariate group and the
        values are keys in :attr:`~anndata.AnnData.uns` storing a dictionary with
        the representation of the covariates. E.g. ``{"drug":"drug_embeddings"}``
        with ``adata.uns["drug_embeddings"] = {"drugA": np.array, "drugB": np.array}``.
    sample_covariates
        Keys in :attr:`~anndata.AnnData.obs` indicating sample covatiates to be taken
        into account for training and prediction, e.g. ``["age", "cell_type"]``.
    sample_covariate_reps
        A dictionary where the keys indicate the name of the covariate group and the
        values are keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the
        representation of the covariates. E.g. ``{"cell_type": "cell_type_embeddings"}`` with
        ``adata.uns["cell_type_embeddings"] = {"cell_typeA": np.array, "cell_typeB": np.array}``.
    split_covariates
        Covariates in :attr:`~anndata.AnnData.obs` to split all control cells into
        different control populations. The perturbed cells are also split according to these
        columns, but if these covariates should also be encoded in the model, the corresponding
        column should also be used in ``perturbation_covariates`` or ``sample_covariates``.
    null_value
        Value to use for padding to ``max_combination_length``.
    primary_group
        Name of the primary perturbation group. If not provided, the first group in
        ``perturbation_covariates`` but ``perturbation_covariates`` should be an ordered dict.
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        sample_rep: str | dict[str, str],
        control_key: str,
        perturbation_covariates: dict[str, Sequence[str]] | None = None,
        perturbation_covariate_reps: dict[str, str] | None = None,
        sample_covariates: Sequence[str] | None = None,
        sample_covariate_reps: dict[str, str] | None = None,
        split_covariates: Sequence[str] | None = None,
        max_combination_length: int | None = None,
        null_value: float = 0.0,
        primary_group: str | None = None,
    ):
        if primary_group is None:
            self.primary_group, _ = next(iter(perturbation_covariates.items()))
        else:
            self.primary_group = primary_group
        self._adata = adata
        self._sample_rep = sample_rep.copy() if isinstance(sample_rep, dict) else sample_rep
        self._sample_rep = self._verify_sample_rep(sample_rep)
        self._control_key = control_key
        self._perturbation_covariates = OrderedDict(self._verify_perturbation_covariates(perturbation_covariates))
        self._perturbation_covariate_reps = OrderedDict(
            self._verify_perturbation_covariate_reps(
                adata,
                perturbation_covariate_reps,
                self._perturbation_covariates,
            )
        )
        self._sample_covariates = self._verify_sample_covariates(sample_covariates)
        self._sample_covariate_reps = self._verify_sample_covariate_reps(
            adata, sample_covariate_reps, self._sample_covariates
        )
        self._split_covariates = self._verify_split_covariates(adata, split_covariates, control_key)
        self._max_combination_length = self._get_max_combination_length(
            self._perturbation_covariates, max_combination_length
        )
        self._null_value = null_value
        self._primary_one_hot_encoder, self._is_categorical = self._get_primary_covar_encoder(
            self._adata,
            self._perturbation_covariates,
            self._perturbation_covariate_reps,
        )
        self._linked_perturb_covars = self._get_linked_perturbation_covariates(
            self._perturbation_covariates, self.primary_group
        )
        sample_cov_groups = OrderedDict({covar: _to_list(covar) for covar in self._sample_covariates})
        covariate_groups = self._perturbation_covariates | sample_cov_groups
        self._covariate_reps = (self._perturbation_covariate_reps or {}) | (self._sample_covariate_reps or {})
        self._covar_to_idx = self._get_covar_to_idx(covariate_groups)  # type: ignore[arg-type]
        perturb_covar_keys = _flatten_list(self._perturbation_covariates.values()) + list(self._sample_covariates)
        perturb_covar_keys += [col for col in self._split_covariates if col not in perturb_covar_keys]
        self._perturb_covar_keys = [k for k in perturb_covar_keys if k is not None]
        self.condition_keys = sorted(self._perturb_covar_keys)
        assert (len(self._perturbation_covariates) > 0) or (len(self._sample_covariates) > 0), (
            "At least one perturbation or sample covariate must be provided."
        )

    def get_train_data(self, adata: anndata.AnnData) -> Any:
        """Get training data for the model.

        Parameters
        ----------
        adata
            An :class:`~anndata.AnnData` object.

        Returns
        -------
        Training data for the model.
        """
        cond_data = self._get_condition_data(adata=adata)
        cell_data = self._get_cell_data(adata)
        return TrainingData(
            cell_data=cell_data,
            split_covariates_mask=cond_data.split_covariates_mask,
            split_idx_to_covariates=cond_data.split_idx_to_covariates,
            perturbation_covariates_mask=cond_data.perturbation_covariates_mask,
            perturbation_idx_to_covariates=cond_data.perturbation_idx_to_covariates,
            perturbation_idx_to_id=cond_data.perturbation_idx_to_id,
            condition_data=cond_data.condition_data,
            control_to_perturbation=cond_data.control_to_perturbation,
            max_combination_length=cond_data.max_combination_length,
            null_value=self._null_value,
            data_manager=self,
        )

    def get_validation_data(
        self,
        adata: anndata.AnnData,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
    ) -> ValidationData:
        """Get validation data for the model.

        Parameters
        ----------
        adata
            An :class:`~anndata.AnnData` object.
        n_conditions_on_log_iteration
            Number of conditions to validate on during logging.
        n_conditions_on_train_end
            Number of conditions to validate on at the end of training.

        Returns
        -------
        Validation data for the model.
        """
        cond_data = self._get_condition_data(adata=adata)
        cell_data = self._get_cell_data(adata)
        return ValidationData(
            cell_data=cell_data,
            split_covariates_mask=cond_data.split_covariates_mask,
            split_idx_to_covariates=cond_data.split_idx_to_covariates,
            perturbation_covariates_mask=cond_data.perturbation_covariates_mask,
            perturbation_idx_to_covariates=cond_data.perturbation_idx_to_covariates,
            perturbation_idx_to_id=cond_data.perturbation_idx_to_id,
            condition_data=cond_data.condition_data,
            control_to_perturbation=cond_data.control_to_perturbation,
            max_combination_length=cond_data.max_combination_length,
            null_value=self._null_value,
            data_manager=self,
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )

    def get_prediction_data(
        self,
        adata: anndata.AnnData,
        sample_rep: str,
        covariate_data: pd.DataFrame,
        rep_dict: dict[str, Any] | None = None,
        condition_id_key: str | None = None,
    ) -> Any:
        """Get predictions for control cells.

        Extracts source distributions from  ``'adata'`` and simulates cells perturbed
        with covariates defined in ``'covariate_data'``.

        Parameters
        ----------
        adata
            An :class:`~anndata.AnnData` object to extract control cells from.
        sample_rep
            Key in :attr:`~anndata.AnnData.obsm` where the sample representation of the control
            is stored or ``'X'`` to use :attr:`~anndata.AnnData.X`.
        covariate_data
            A :class:`~pandas.DataFrame` with columns defining the covariates as
            in :meth:`cellflow.model.CellFlow.prepare_data` and stored in
            :attr:`cellflow.model.CellFlow.data_manager`.
        rep_dict
            Dictionary with representations of the covariates.
            If not provided, :attr:`~anndata.AnnData.uns` is used.
        condition_id_key
            Key in :class:`~pandas.DataFrame` that defines the condition names.

        Returns
        -------
        Training data for the model.
        """
        self._verify_prediction_data(adata)
        split_cov_combs = self._get_split_cov_combs(covariate_data=covariate_data)

        # adata is None since we don't extract cell masks for predicted covariates
        cond_data = self._get_condition_data(
            adata=None,
            covariate_data=covariate_data,
            rep_dict=adata.uns if rep_dict is None else rep_dict,
            condition_id_key=condition_id_key,
        )
        cell_data = self._get_cell_data(adata, sample_rep)
        split_covariates_mask, split_idx_to_covariates = self._get_split_covariates_mask(
            adata=adata, split_cov_combs=split_cov_combs
        )

        return PredictionData(
            cell_data=cell_data,
            split_covariates_mask=split_covariates_mask,
            split_idx_to_covariates=split_idx_to_covariates,
            condition_data=cond_data.condition_data,
            control_to_perturbation=cond_data.control_to_perturbation,
            perturbation_idx_to_covariates=cond_data.perturbation_idx_to_covariates,
            perturbation_idx_to_id=cond_data.perturbation_idx_to_id,
            max_combination_length=cond_data.max_combination_length,
            null_value=self._null_value,
            data_manager=self,
        )

    def get_condition_data(
        self,
        covariate_data: pd.DataFrame,
        rep_dict: dict[str, Any] | None = None,
        condition_id_key: str | None = None,
    ) -> ConditionData:
        """Get condition data for the model.

        Parameters
        ----------
        covariate_data
            Dataframe with covariates.
        condition_id_key
            Key in ``covariate_data`` that defines the condition id.
        rep_dict

        Returns
        -------
        Condition data for the model.
        """
        self._verify_covariate_data(covariate_data, self._perturb_covar_keys)
        cond_data = self._get_condition_data(
            adata=None,
            covariate_data=covariate_data,
            rep_dict=rep_dict,
            condition_id_key=condition_id_key,
        )
        return ConditionData(
            condition_data=cond_data.condition_data,
            max_combination_length=cond_data.max_combination_length,
            perturbation_idx_to_covariates=cond_data.perturbation_idx_to_covariates,
            perturbation_idx_to_id=cond_data.perturbation_idx_to_id,
            null_value=self._null_value,
            data_manager=self,
        )

    def _get_split_cov_combs(self, covariate_data: pd.DataFrame) -> np.ndarray | list[list[Any]]:
        if len(self._split_covariates) > 0:
            sorted_df = covariate_data[self._split_covariates].drop_duplicates().sort_values(by=self._split_covariates)
            res = [list(row) for row in sorted_df.values]
            return res
        else:
            return [[]]

    @staticmethod
    def _get_embeddings(
        condition_data: pd.DataFrame,
        rep_dict: dict[str, dict[str, ArrayLike]],
        perturb_covariates: Any,
        covariate_reps: dict[str, str],
        is_categorical: bool,
        primary_one_hot_encoder: preprocessing.OneHotEncoder | None,
        null_value: float,
        max_combination_length: int,
        linked_perturb_covars: dict[str, dict[Any, Any]],
        sample_covariates: Sequence[str],
        primary_group: str,
    ) -> dict[str, np.ndarray]:
        perturb_covar_emb = DataManager._get_perturbation_covariates_embeddings(
            condition_data=condition_data,
            rep_dict=rep_dict,
            perturb_covariates=perturb_covariates,
            covariate_reps=covariate_reps,
            is_categorical=is_categorical,
            primary_one_hot_encoder=primary_one_hot_encoder,
            null_value=null_value,
            max_combination_length=max_combination_length,
            linked_perturb_covars=linked_perturb_covars,
            primary_group=primary_group,
        )
        sample_covar_emb = DataManager._get_sample_covariates_embedding(
            condition_data=condition_data,
            rep_dict=rep_dict,
            sample_covariates=sample_covariates,
            covariate_reps=covariate_reps,
            max_combination_length=max_combination_length,
        )
        return perturb_covar_emb | sample_covar_emb

    @staticmethod
    def _get_perturbation_covariates_embeddings(
        condition_data: pd.DataFrame,
        rep_dict: dict[str, dict[str, ArrayLike]],
        perturb_covariates: Any,
        covariate_reps: dict[str, str],
        is_categorical: bool,
        primary_one_hot_encoder: preprocessing.OneHotEncoder | None,
        null_value: float,
        max_combination_length: int,
        linked_perturb_covars: dict[str, dict[Any, Any]],
        primary_group: str,
    ) -> dict[str, np.ndarray]:
        """Get perturbation covariates embeddings.

        Parameters
        ----------
        condition_data
            DataFrame with condition data.
        rep_dict
            Dictionary with representations of covariates.
        perturb_covariates
            Dictionary with perturbation covariates.
        covariate_reps
            Dictionary with representations of covariates.
        is_categorical
            Whether primary covariate is categorical.
        primary_one_hot_encoder
            One-hot encoder for primary covariates.
        null_value
            Value to use for padding.
        max_combination_length
            Maximum combination length of perturbation covariates.
        linked_perturb_covars
            Dictionary linking primary covariates to other covariates.
        sample_covariates
            Sample covariates.

        Returns
        -------
        Dictionary with perturbation covariate embeddings.
        """
        perturb_covar_emb: dict[str, list[np.ndarray]] = {group: [] for group in perturb_covariates}
        primary_covars = perturb_covariates[primary_group]
        for primary_cov in primary_covars:
            value = condition_data[primary_cov]
            cov_name = value if is_categorical else primary_cov  # drug a
            prim_arr1 = None
            if primary_group in covariate_reps:
                rep_key = covariate_reps[primary_group]
                if cov_name not in rep_dict[rep_key]:
                    raise ValueError(f"Representation for '{cov_name}' not found in `adata.uns['{rep_key}']`.")
                prim_arr1 = np.asarray(rep_dict[rep_key][cov_name])
            else:
                prim_arr1 = np.asarray(
                    primary_one_hot_encoder.transform(  # type: ignore[union-attr]
                        np.array(cov_name).reshape(-1, 1)
                    )
                )
            prim_arr2 = None
            if not is_categorical:
                prim_arr2 = value * prim_arr1.copy()
            else:
                prim_arr2 = prim_arr1.copy()

            prim_arr3 = DataManager._check_shape(prim_arr2.copy())
            perturb_covar_emb[primary_group].append(prim_arr3)

            for linked_covar in linked_perturb_covars[primary_cov].items():
                linked_group, linked_cov = list(linked_covar)

                if linked_cov is None:
                    linked_arr = np.full((1, 1), null_value)
                    linked_arr = DataManager._check_shape(linked_arr)
                    perturb_covar_emb[linked_group].append(linked_arr)
                    continue

                cov_name = condition_data[linked_cov]

                if linked_group in covariate_reps:
                    rep_key = covariate_reps[linked_group]
                    if cov_name not in rep_dict[rep_key]:
                        raise ValueError(f"Representation for '{cov_name}' not found in `adata.uns['{linked_group}']`.")
                    linked_arr = np.asarray(rep_dict[rep_key][cov_name])
                else:
                    linked_arr = np.asarray(condition_data[linked_cov])
                linked_arr = DataManager._check_shape(linked_arr)
                perturb_covar_emb[linked_group].append(linked_arr)

        perturb_covar_emb = {
            k: DataManager._pad_to_max_length(
                np.concatenate(v, axis=0),
                max_combination_length,
                null_value,
            )
            for k, v in perturb_covar_emb.items()
        }
        return perturb_covar_emb

    @staticmethod
    def _get_perturb_covar_df(
        covariate_data: pd.DataFrame, perturb_covar_keys: list[str], condition_id_key: str | None
    ) -> pd.DataFrame:
        # Extract unique combinations of perturbation covariates
        select_keys = perturb_covar_keys.copy()
        if condition_id_key is not None:
            select_keys += [condition_id_key]
        perturb_covar_df = covariate_data[select_keys].drop_duplicates()
        if condition_id_key is not None:
            perturb_covar_df = perturb_covar_df.set_index(condition_id_key)
        else:
            perturb_covar_df = perturb_covar_df.reset_index()
        return perturb_covar_df

    @staticmethod
    def _get_sample_covariates_embedding(
        condition_data: pd.DataFrame,
        rep_dict: dict[str, dict[str, ArrayLike]],
        sample_covariates: Sequence[str],
        covariate_reps: dict[str, str],
        max_combination_length: int,
    ) -> dict[str, np.ndarray]:
        sample_covar_emb: dict[str, np.ndarray] = {}
        for sample_cov in sample_covariates:
            value = condition_data[sample_cov]
            if sample_cov in covariate_reps:
                rep_key = covariate_reps[sample_cov]

                if value not in rep_dict[rep_key]:
                    raise ValueError(f"Representation for '{value}' not found in `adata.uns['{sample_cov}']`.")
                cov_arr = np.asarray(rep_dict[rep_key][value])
            else:
                cov_arr = np.asarray(value)
            cov_arr = DataManager._check_shape(cov_arr)
            sample_covar_emb[sample_cov] = np.tile(cov_arr, (max_combination_length, 1))

        return sample_covar_emb

    def _get_pert_data(
        self,
        df: pd.DataFrame,
        all_combs_keys: list[str],
        perturbation_covariates_keys: list[str],
        uniq_sample_keys: list[str],
        orig_cell_idx: list[int],
        train: bool,
    ):
        """
        Helper function to be used in _get_condition_data.

        If train is False, i.e. all cells are control cells then the output is:
        - control_to_perturbation: {0: [perturbation_covariates_to_idx.values()]}
        - perturbation_idx_to_covariates: idxs only determined by unique values of self.perturb_covar_keys
        - perturbation_covariates_to_idx: inverse of perturbation_idx_to_covariates
        - tgt_conds_df: pd.DataFrame with unique values of all_combs_keys that are all control cells
        - df: pd.DataFrame with all_combs_keys and control_key

        If train is True, i.e. there are perturbation cells then the output is:
        - control_to_perturbation: {control_idx: [perturbation_idx]}
        - perturbation_idx_to_covariates: {perturbation_idx: [perturbation_covariates_keys]}
        - perturbation_idx_to_id: inverse of perturbation_idx_to_covariates
        - tgt_conds_df: pd.DataFrame with unique values of all_combs_keys without control cells
        - df reindexed to orig_cell_idx (i.e. covariate_data.index)
        """
        if train:
            tgt_conds_df = (
                df[~df[self.control_key]][all_combs_keys].sort_values(by=all_combs_keys).drop_duplicates(keep="first")
            )
            perturbation_idx_to_covariates = (
                df[["global_pert_mask", *all_combs_keys]].groupby(["global_pert_mask"]).first().to_dict(orient="index")
            )
            perturbation_idx_to_covariates = {
                int(k): [v[s] for s in [*perturbation_covariates_keys, *uniq_sample_keys]]
                for k, v in perturbation_idx_to_covariates.items()
            }
            perturbation_covariates_to_idx = {tuple(v): k for k, v in perturbation_idx_to_covariates.items()}

            control_to_perturbation = (
                df[~df[self.control_key]].groupby(["global_control_mask"])["global_pert_mask"].unique()
            )
            control_to_perturbation = control_to_perturbation.to_dict()
            control_to_perturbation = {
                k: np.array(sorted(v), dtype=np.int32) for k, v in control_to_perturbation.items()
            }
            df.set_index("cell_index", inplace=True)
            df = df.reindex(orig_cell_idx)
        else:
            tgt_conds_df = df[all_combs_keys].sort_values(by=self.perturb_covar_keys).drop_duplicates(keep="first")
            p = tgt_conds_df[self.perturb_covar_keys].drop_duplicates()
            p.sort_values(by=self.perturb_covar_keys, inplace=True)
            p.index = np.arange(len(p))
            perturbation_idx_to_covariates = {int(p.index[i]): tuple(p.iloc[i]) for i in range(len(p))}
            perturbation_covariates_to_idx = {tuple(v): k for k, v in perturbation_idx_to_covariates.items()}
            control_to_perturbation = {
                0: sorted(perturbation_covariates_to_idx.values()),
            }

        return (
            perturbation_idx_to_covariates,
            perturbation_covariates_to_idx,
            control_to_perturbation,
            tgt_conds_df,
            df,
        )

    def _get_condition_data(
        self,
        adata: anndata.AnnData | None,
        covariate_data: pd.DataFrame | None = None,
        rep_dict: dict[str, Any] | None = None,
        condition_id_key: str | None = None,
    ) -> ReturnData:
        # for training/validation: adata is provided and used to get cell masks, covariate_data is None
        if adata is None and covariate_data is None:
            raise ValueError("Either `adata` or `covariate_data` must be provided.")
        covariate_data = covariate_data if covariate_data is not None else adata.obs  # type: ignore[union-attr]
        if rep_dict is None:
            rep_dict = adata.uns if adata is not None else {}
        control_to_perturbation: dict[int, ArrayLike] = {}
        split_idx_to_covariates: dict[int, tuple[Any]] = {}
        perturbation_idx_to_covariates: dict[int, tuple[Any]] = {}
        perturbation_idx_to_id: dict[int, Any] = {}
        split_covariates_mask = None
        perturbation_covariates_mask = None
        condition_data: dict[str, list[np.ndarray]] = {i: [] for i in self._covar_to_idx.keys()}
        perturb_covariates = OrderedDict({k: sorted(_to_list(v)) for k, v in self.perturbation_covariates.items()})
        npartitions = 2  # TODO: make this dynamic

        return_mask: bool = not (
            adata is None and (len(self.split_covariates) == 0 and len(self.sample_covariates) == 0)
        )

        # check if all perturbation/split covariates and control cells are present in the input
        self._verify_control_data(adata)
        self._verify_covariate_data(covariate_data, self.sample_covariates)
        self._verify_covariate_data(covariate_data, self.split_covariates)
        if condition_id_key is not None:
            self._verify_condition_id_key(covariate_data, condition_id_key)

        # extract unique combinations of perturbation covariates
        perturb_covar_df = DataManager._get_perturb_covar_df(covariate_data, self.perturb_covar_keys, condition_id_key)

        covariate_data = covariate_data.copy()
        covariate_data["cell_index"] = covariate_data.index
        covariate_data = covariate_data.reset_index(drop=True)

        uniq_sample_keys = self.split_covariates
        if len(self.split_covariates) == 0:
            uniq_sample_keys = self.sample_covariates

        perturbation_covariates_keys = [key for key in self.perturb_covar_keys if key not in uniq_sample_keys]

        all_combs_keys = uniq_sample_keys + perturbation_covariates_keys
        if len(self.split_covariates) == 0:
            all_combs_keys = perturbation_covariates_keys + uniq_sample_keys

        df = covariate_data[uniq_sample_keys + perturbation_covariates_keys + [self.control_key]].copy()
        cell_idx_key = "cell_index"
        df[cell_idx_key] = df.index
        df = df.set_index(cell_idx_key, drop=False)
        for col in all_combs_keys:
            if df[col].dtype != "category":
                df[col] = df[col].astype("category")
        ddf = dd.from_pandas(df, npartitions=npartitions)
        ddf = ddf.sort_values(by=[*uniq_sample_keys, *perturbation_covariates_keys, self.control_key])
        ddf = ddf.reset_index(drop=True)

        all_combs = ddf[uniq_sample_keys + perturbation_covariates_keys + [self.control_key]].drop_duplicates(
            keep="first", subset=uniq_sample_keys + perturbation_covariates_keys + [self.control_key]
        )
        control_combs = all_combs[uniq_sample_keys + [self.control_key]].drop_duplicates(
            keep="first", subset=uniq_sample_keys + [self.control_key]
        )
        with ProgressBar():
            control_combs, all_combs, df = dask.compute(control_combs, all_combs, ddf)

        control_combs = control_combs[control_combs[self.control_key]].sort_values(by=uniq_sample_keys)
        all_combs = all_combs[~all_combs[self.control_key]].sort_values(by=all_combs_keys)

        all_combs["global_pert_mask"] = np.arange(len(all_combs), dtype=np.int64)
        control_combs["global_control_mask"] = np.arange(len(control_combs), dtype=np.int64)

        control_combs = control_combs.sort_values(by=uniq_sample_keys)
        all_combs = all_combs.sort_values(by=all_combs_keys)

        all_combs = all_combs.drop(columns=[self.control_key])
        control_combs = control_combs.drop(columns=[self.control_key])

        if len(self.split_covariates) > 0:
            df = df.merge(control_combs, on=uniq_sample_keys, how="left")
        else:
            df["global_control_mask"] = 0

        # Then merge with all_combs
        df = df.merge(
            all_combs,
            on=uniq_sample_keys + perturbation_covariates_keys,
            how="left",
        )

        df = df.sort_values(by=all_combs_keys)

        df["split_covariates_mask"] = df["global_control_mask"]
        df.loc[~df[self.control_key], "split_covariates_mask"] = -1
        df["split_covariates_mask"] = df["split_covariates_mask"].astype(np.int64)

        df["perturbation_covariates_mask"] = df["global_pert_mask"]
        df.loc[df[self.control_key], "perturbation_covariates_mask"] = -1
        df["perturbation_covariates_mask"] = df["perturbation_covariates_mask"].astype(np.int64)

        all_control: bool = df[self.control_key].all() or (adata is None)

        perturbation_idx_to_covariates, perturbation_covariates_to_idx, control_to_perturbation, tgt_conds_df, df = (
            self._get_pert_data(
                df=df,
                all_combs_keys=all_combs_keys,
                perturbation_covariates_keys=perturbation_covariates_keys,
                uniq_sample_keys=uniq_sample_keys,
                orig_cell_idx=covariate_data.index,
                train=not all_control,
            )
        )

        if return_mask:
            split_idx_to_covariates = (
                df[["global_control_mask", *self.split_covariates]]
                .groupby(["global_control_mask"])
                .first()
                .to_dict(orient="index")
            )
            split_idx_to_covariates = {
                k: tuple(v[s] for s in self.split_covariates) for k, v in split_idx_to_covariates.items()
            }
            split_covariates_mask = np.asarray(df["split_covariates_mask"].values, dtype=np.int32)
            perturbation_covariates_mask = np.asarray(df["perturbation_covariates_mask"].values, dtype=np.int32)

        if condition_id_key is not None:
            perturb_covar_df.reset_index(names="_condition_id", inplace=True)
            perturb_covar_df.set_index(self.perturb_covar_keys, inplace=True)

        delayed_results = []

        def _process_cond_wrapper(tgt_idx, tgt_cond):
            embeddings = DataManager._get_embeddings(
                condition_data=tgt_cond,
                rep_dict=rep_dict,
                perturb_covariates=perturb_covariates,
                covariate_reps=self.covariate_reps,
                is_categorical=self.is_categorical,
                primary_one_hot_encoder=self.primary_one_hot_encoder,
                null_value=self.null_value,
                max_combination_length=self.max_combination_length,
                linked_perturb_covars=self.linked_perturb_covars,
                sample_covariates=self.sample_covariates,
                primary_group=self.primary_group,
            )
            return tgt_idx, embeddings

        for _, tgt_cond in tgt_conds_df.iterrows():
            tgt_idx = perturbation_covariates_to_idx[tuple(tgt_cond[perturbation_covariates_keys + uniq_sample_keys])]
            tgt_cond = tgt_cond[self.perturb_covar_keys]
            if condition_id_key is not None:
                perturbation_idx_to_id[tgt_idx] = perturb_covar_df.loc[tuple(tgt_cond)]["_condition_id"]
            tgt_cond = dict(tgt_cond)
            delayed_results.append(dask.delayed(_process_cond_wrapper)(tgt_idx=tgt_idx, tgt_cond=tgt_cond))
        with ProgressBar():
            results = dask.compute(*delayed_results)
        results = sorted(results, key=lambda x: x[0])
        for _, embeddings in results:
            for pert_cov, emb in embeddings.items():
                condition_data[pert_cov].append(emb)

        for pert_cov, emb in condition_data.items():
            condition_data[pert_cov] = np.array(emb)

        res = ReturnData(
            split_covariates_mask=split_covariates_mask,
            split_idx_to_covariates=split_idx_to_covariates,
            perturbation_covariates_mask=perturbation_covariates_mask,
            perturbation_idx_to_covariates=perturbation_idx_to_covariates,
            perturbation_idx_to_id=perturbation_idx_to_id,
            condition_data=condition_data,  # type: ignore[arg-type]
            control_to_perturbation=control_to_perturbation,
            max_combination_length=self._max_combination_length,
        )
        return res

    @staticmethod
    def _verify_condition_id_key(covariate_data: pd.DataFrame, condition_id_key: str | None) -> None:
        if condition_id_key is not None and condition_id_key not in covariate_data.columns:
            raise ValueError(f"The `condition_id_key` column ('{condition_id_key}') was not found in `adata.obs`.")
        if not len(covariate_data[condition_id_key].unique()) == len(covariate_data):
            raise ValueError(f"The `condition_id_key` column ('{condition_id_key}') must contain unique values.")

    @staticmethod
    def _verify_sample_rep(sample_rep: str | dict[str, str]) -> str | dict[str, str]:
        if not (isinstance(sample_rep, str) or isinstance(sample_rep, dict)):
            raise ValueError(
                f"`sample_rep` should be of type `str` or `dict`, found {sample_rep} to be of type {type(sample_rep)}."
            )
        return sample_rep

    def _get_cell_data(
        self,
        adata: anndata.AnnData,
        sample_rep: str | None = None,
    ) -> np.ndarray:
        sample_rep = self._sample_rep if sample_rep is None else sample_rep
        if sample_rep == "X":
            sample_rep = adata.X
            if isinstance(sample_rep, sp.csr_matrix):
                return np.asarray(sample_rep.toarray())
            else:
                return np.asarray(sample_rep)
        if isinstance(self._sample_rep, str):
            if self._sample_rep not in adata.obsm:
                raise KeyError(f"Sample representation '{self._sample_rep}' not found in `adata.obsm`.")
            return np.asarray(adata.obsm[self._sample_rep])
        attr, key = next(iter(sample_rep.items()))  # type: ignore[union-attr]
        return np.asarray(getattr(adata, attr)[key])

    def _verify_control_data(self, adata: anndata.AnnData | None) -> None:
        if adata is None:
            return None
        if self._control_key not in adata.obs:
            raise ValueError(f"Control column '{self._control_key}' not found in adata.obs.")
        if not isinstance(adata.obs[self._control_key].dtype, pd.BooleanDtype):
            try:
                adata.obs[self._control_key] = adata.obs[self._control_key].astype("boolean")
            except TypeError as e:
                raise ValueError(f"Control column '{self._control_key}' could not be converted to boolean.") from e
        if adata.obs[self._control_key].sum() == 0:
            raise ValueError("No control cells found in adata.")

    def _verify_prediction_data(self, adata: anndata.AnnData) -> None:
        if self._control_key not in adata.obs:
            raise ValueError(f"Control column '{self._control_key}' not found in adata.obs.")
        if not isinstance(adata.obs[self._control_key].dtype, pd.BooleanDtype):
            try:
                adata.obs[self._control_key] = adata.obs[self._control_key].astype("boolean")
            except ValueError as e:
                raise ValueError(f"Control column '{self._control_key}' could not be converted to boolean.") from e
        if not adata.obs[self._control_key].all():
            raise ValueError(
                f"For prediction, all cells in `adata` should be from control condition. Ensure that '{self._control_key}' is `True` for all cells, even if you're setting `.obs` to predicted condition."
            )

    def _get_split_combination_mask(
        self,
        covariate_data: pd.DataFrame,
        split_covariates_mask: ArrayLike,
        split_combination: ArrayLike,
        split_idx_to_covariates: dict[int, tuple[Any]],
        control_mask: ArrayLike,
        src_counter: int,
    ) -> tuple[ArrayLike, dict[int, tuple[Any]], ArrayLike]:
        filter_dict = dict(zip(self.split_covariates, split_combination, strict=False))
        split_cov_mask = (covariate_data[list(filter_dict.keys())] == list(filter_dict.values())).all(axis=1)
        mask = np.array(control_mask * split_cov_mask).astype(bool)
        split_covariates_mask[mask] = src_counter
        split_idx_to_covariates[src_counter] = tuple(split_combination)
        return split_covariates_mask, split_idx_to_covariates, split_cov_mask

    def _get_split_covariates_mask(
        self, adata: anndata.AnnData, split_cov_combs: np.ndarray | list[list[Any]]
    ) -> tuple[ArrayLike, dict[int, tuple[Any]]]:
        # here we assume that adata only contains source cells
        if len(self.split_covariates) == 0:
            return np.full((len(adata),), 0, dtype=np.int32), {}
        split_covariates_mask = np.full((len(adata),), -1, dtype=np.int32)
        split_idx_to_covariates: dict[int, Any] = {}
        src_counter = 0
        for split_combination in split_cov_combs:
            split_covariates_mask_previous = split_covariates_mask.copy()
            split_covariates_mask, split_idx_to_covariates, _ = self._get_split_combination_mask(
                covariate_data=adata.obs,
                split_covariates_mask=split_covariates_mask,
                split_combination=split_combination,
                split_idx_to_covariates=split_idx_to_covariates,
                control_mask=np.ones((adata.n_obs,)),
                src_counter=src_counter,
            )

            if (split_covariates_mask == split_covariates_mask_previous).all():
                raise ValueError(f"No cells found in `adata` for split covariates {split_combination}.")
            src_counter += 1
        return np.asarray(split_covariates_mask), split_idx_to_covariates

    @staticmethod
    def _verify_perturbation_covariates(data: dict[str, Sequence[str]] | None) -> dict[str, list[str]]:
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(
                f"`perturbation_covariates` should be a dictionary, found {data} to be of type {type(data)}."
            )
        if len(data) == 0:
            raise ValueError("No perturbation covariates provided.")
        for key, covars in data.items():
            if not isinstance(key, str):
                raise ValueError(f"Key should be a string, found {key} to be of type {type(key)}.")
            if not isinstance(covars, tuple | list):
                raise ValueError(f"Value should be a tuple, found {covars} to be of type {type(covars)}.")
            if len(covars) == 0:
                raise ValueError(f"No covariates provided for perturbation group {key}.")
        lengths = [len(covs) for covs in data.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f"Length of perturbation covariate groups must match, found lengths {lengths}.")
        return {k: sorted(el) for k, el in data.items()}

    @staticmethod
    def _verify_sample_covariates(
        sample_covariates: Sequence[str] | None,
    ) -> list[str]:
        if sample_covariates is None:
            return []
        if not isinstance(sample_covariates, tuple | list):
            raise ValueError(
                f"`sample_covariates` should be a tuple or list, found {sample_covariates} to be of type {type(sample_covariates)}."
            )
        for covar in sample_covariates:
            if not isinstance(covar, str):
                raise ValueError(f"Key should be a string, found {covar} to be of type {type(covar)}.")
        return sorted(sample_covariates)

    @staticmethod
    def _verify_split_covariates(
        adata: anndata.AnnData,
        data: Sequence[str] | None,
        control_key: str,
    ) -> Sequence[str]:
        if data is None:
            return []
        if not isinstance(data, tuple | list):
            raise ValueError(f"`split_covariates` should be a tuple or list, found {data} to be of type {type(data)}.")
        for covar in data:
            if not isinstance(covar, str):
                raise ValueError(f"Key should be a string, found {covar} to be of type {type(covar)}.")
        source_splits = adata.obs[adata.obs[control_key]][data].drop_duplicates()
        source_splits = map(tuple, source_splits.values)
        target_splits = adata.obs[~adata.obs[control_key]][data].drop_duplicates()
        target_splits = map(tuple, target_splits.values)
        source_without_targets = set(source_splits) - set(target_splits)
        if len(source_without_targets) > 0:
            raise ValueError(
                f"Source distribution with split covariate values {source_without_targets} do not have a corresponding target distribution."
            )
        return sorted(data)

    @staticmethod
    def _verify_covariate_data(covariate_data: pd.DataFrame, covars) -> None:
        for covariate in covars:
            if covariate is not None and covariate not in covariate_data:
                raise ValueError(f"Covariate {covariate} not found in adata.obs or covariate_data.")

    @staticmethod
    def _get_linked_perturbation_covariates(
        perturb_covariates: dict[str, list[str]],
        primary_group: str,
    ) -> dict[str, dict[Any, Any]]:
        perturb_covariates = OrderedDict(perturb_covariates)
        primary_covars = perturb_covariates[primary_group]
        linked_perturb_covars: dict[str, dict[Any, Any]] = {k: {} for k in primary_covars}
        for cov_group, covars in list(perturb_covariates.items())[1:]:
            for primary_cov, linked_cov in zip(primary_covars, covars, strict=False):
                linked_perturb_covars[primary_cov][cov_group] = linked_cov

        return OrderedDict(linked_perturb_covars)

    @staticmethod
    def _verify_perturbation_covariate_reps(
        adata: anndata.AnnData,
        perturbation_covariate_reps: dict[str, str] | None,
        perturbation_covariates: dict[str, list[str]],
    ) -> dict[str, str]:
        if perturbation_covariate_reps is None:
            return {}
        for key, value in perturbation_covariate_reps.items():
            if key not in perturbation_covariates:
                raise ValueError(f"Key '{key}' not found in covariates.")
            if value not in adata.uns:
                raise ValueError(f"Perturbation covariate representation '{value}' not found in `adata.uns`.")
            if not isinstance(adata.uns[value], dict):
                raise ValueError(
                    f"Perturbation covariate representation '{value}' in `adata.uns` should be of type `dict`, found {type(adata.uns[value])}."
                )
        return perturbation_covariate_reps

    @staticmethod
    def _verify_sample_covariate_reps(
        adata: anndata.AnnData,
        sample_covariate_reps: dict[str, str] | None,
        covariates: list[str],
    ) -> dict[str, str]:
        if sample_covariate_reps is None:
            return {}
        for key, value in sample_covariate_reps.items():
            if key not in covariates:
                raise ValueError(f"Key '{key}' not found in covariates.")
            if value not in adata.uns:
                raise ValueError(f"Sample covariate representation '{value}' not found in `adata.uns`.")
            if not isinstance(adata.uns[value], dict):
                raise ValueError(
                    f"Sample covariate representation '{value}' in `adata.uns` should be of type `dict`, found {type(adata.uns[value])}."
                )
        return OrderedDict(sample_covariate_reps)

    @staticmethod
    def _get_max_combination_length(
        perturbation_covariates: dict[str, list[str]],
        max_combination_length: int | None,
    ) -> int:
        obs_max_combination_length = max(len(comb) for comb in perturbation_covariates.values())
        if max_combination_length is None:
            return obs_max_combination_length
        elif max_combination_length < obs_max_combination_length:
            logger.warning(
                f"Provided `max_combination_length` is smaller than the observed maximum combination length of the perturbation covariates. Setting maximum combination length to {obs_max_combination_length}.",
                stacklevel=2,
            )
            return obs_max_combination_length
        else:
            return max_combination_length

    def _get_primary_covar_encoder(
        self,
        adata: anndata.AnnData,
        perturbation_covariates: dict[str, list[str]],
        perturbation_covariate_reps: dict[str, str],
    ) -> tuple[preprocessing.OneHotEncoder | None, bool]:
        primary_covars = perturbation_covariates[self.primary_group]
        is_categorical = self._check_covariate_type(adata, primary_covars)
        if perturbation_covariate_reps and self.primary_group in perturbation_covariate_reps:
            return None, is_categorical
        if is_categorical:
            encoder = preprocessing.OneHotEncoder(sparse_output=False)
            all_values = np.unique(adata.obs[primary_covars].values.flatten())
            encoder.fit(all_values.reshape(-1, 1))
            return encoder, is_categorical
        encoder = preprocessing.OneHotEncoder(sparse_output=False)
        encoder.fit(np.array(primary_covars).reshape(-1, 1))
        return encoder, is_categorical

    @staticmethod
    def _check_covariate_type(adata: anndata.AnnData, covars: Sequence[str]) -> bool:
        col_is_cat = []
        for covariate in covars:
            if is_numeric_dtype(adata.obs[covariate]):
                col_is_cat.append(False)
                continue
            if adata.obs[covariate].isin(["True", "False", True, False]).all():
                adata.obs[covariate] = adata.obs[covariate].astype(int)
                col_is_cat.append(False)
                continue
            try:
                adata.obs[covariate] = adata.obs[covariate].astype("category")
                col_is_cat.append(True)
            except ValueError as e:
                raise ValueError(
                    f"Perturbation covariates `{covariate}` should be either numeric/boolean or categorical."
                ) from e

        if max(col_is_cat) != min(col_is_cat):
            raise ValueError(
                f"Groups of perturbation covariates `{covariate}` should be either all numeric/boolean or all categorical."
            )

        return max(col_is_cat)

    @staticmethod
    def _verify_covariate_type(covariate_data: pd.DataFrame, covars: Sequence[str], categorical: bool) -> None:
        for covariate in covars:
            if is_numeric_dtype(covariate_data[covariate]):
                if categorical:
                    raise ValueError(f"Perturbation covariates `{covariate}` should be categorical, found numeric.")
                continue
            if covariate_data[covariate].isin(["True", "False", True, False]).all():
                if categorical:
                    raise ValueError(f"Perturbation covariates `{covariate}` should be categorical, found boolean.")
                continue
            try:
                covariate_data[covariate] = covariate_data[covariate].astype("category")
            except ValueError as e:
                raise ValueError(
                    f"Perturbation covariates `{covariate}` should be either numeric/boolean or categorical."
                ) from e
            else:
                if not categorical:
                    raise ValueError(
                        f"Perturbation covariates `{covariate}` should be numeric/boolean, found categorical."
                    )

    @staticmethod
    def _check_shape(arr: float | ArrayLike) -> ArrayLike:
        if not hasattr(arr, "shape") or len(arr.shape) == 0:
            return np.ones((1, 1)) * arr
        if arr.ndim == 1:  # type: ignore[union-attr]
            return np.expand_dims(arr, 0)
        elif arr.ndim == 2:  # type: ignore[union-attr]
            if arr.shape[0] == 1:
                return arr  # type: ignore[return-value]
            if arr.shape[1] == 1:
                return np.transpose(arr)
            raise ValueError(
                "Condition representation has an unexpected shape. Should be (1, n_features) or (n_features, )."
            )
        elif arr.ndim > 2:  # type: ignore[union-attr]
            raise ValueError("Condition representation has too many dimensions. Should be 1 or 2.")

        raise ValueError(
            "Condition representation as an unexpected format. Expected an array of shape (1, n_features) or (n_features, )."
        )

    @staticmethod
    def _get_covar_to_idx(covariate_groups: dict[str, Sequence[str]]) -> dict[str, int]:
        idx_to_covar = {}
        for idx, cov_group in enumerate(covariate_groups):
            idx_to_covar[idx] = cov_group
        covar_to_idx = {v: k for k, v in idx_to_covar.items()}
        return OrderedDict(covar_to_idx)

    @staticmethod
    def _pad_to_max_length(arr: np.ndarray, max_combination_length: int, null_value: Any) -> np.ndarray:
        if arr.shape[0] < max_combination_length:
            null_arr = np.full((max_combination_length - arr.shape[0], arr.shape[1]), null_value)
            arr = np.concatenate([arr, null_arr], axis=0)
        return arr

    @property
    def is_categorical(self) -> bool:
        """Whether the primary covariate is categorical."""
        return self._is_categorical

    @property
    def adata(self) -> anndata.AnnData:
        """An :class:`~anndata.AnnData` object used for instantiating the DataManager."""
        return self._adata

    @property
    def control_key(self) -> str:
        """Boolean key in :attr:`~anndata.AnnData.obs` indicating whether belongs to control group."""
        return self._control_key

    @property
    def perturbation_covariates(self) -> dict[str, list[str]]:
        """Dictionary with keys indicating the name of the covariate group and values are keys in :attr:`~anndata.AnnData.obs` which together define the covariates."""
        return self._perturbation_covariates

    @property
    def perturbation_covariate_reps(self) -> dict[str, str]:
        """Dictionary with keys indicating the name of the covariate group and values are keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the representation of the covariates."""
        return self._perturbation_covariate_reps

    @property
    def sample_covariates(self) -> Sequence[str]:
        """Keys in :attr:`~anndata.AnnData.obs` indicating which sample the cell belongs to (e.g. cell line)."""
        return self._sample_covariates

    @property
    def sample_covariate_reps(self) -> dict[str, str]:
        """Dictionary with keys indicating the name of the sample covariate group and values are keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the representation of the sample covariates."""
        return self._sample_covariate_reps

    @property
    def split_covariates(self) -> Sequence[str]:
        """Covariates in :attr:`~anndata.AnnData.obs` to split all control cells into different control populations."""
        return self._split_covariates

    @property
    def max_combination_length(self) -> int:
        """Maximum combination length of perturbation covariates."""
        return self._max_combination_length

    @property
    def null_value(self) -> float:
        """Value to use for padding to :attr:`~max_combination_length`."""
        return self._null_value

    @property
    def primary_one_hot_encoder(self) -> preprocessing.OneHotEncoder | None:
        """One-hot encoder for the primary covariate."""
        return self._primary_one_hot_encoder

    @property
    def linked_perturb_covars(self) -> dict[str, dict[Any, Any]]:
        """Dictionary with keys indicating the name of the primary covariate and values are dictionaries with keys indicating the name of the linked covariate group and values are the linked covariates."""
        return self._linked_perturb_covars

    @property
    def covariate_reps(self) -> dict[str, str]:
        """Dictionary which stores representation of covariates, i.e. the union of ``sample_covariate_reps`` and ``perturbation_covariate_reps``."""
        return self._covariate_reps

    @property
    def covar_to_idx(self) -> dict[str, int]:
        """TODO: add description"""
        return self._covar_to_idx

    @property
    def perturb_covar_keys(self) -> list[str]:
        """List of all perturbation covariates."""
        return self._perturb_covar_keys

    @property
    def sample_rep(self) -> str | dict[str, str]:
        """Key of the sample representation."""
        return self._sample_rep
