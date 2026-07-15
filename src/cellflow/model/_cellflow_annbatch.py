"""Streaming (annbatch/dagloader) CellFlow: the default path.

Trains, validates and predicts over an out-of-core :class:`~annbatch.DatasetCollection` or an
in-memory ``AnnData`` alike.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np
import pandas as pd

from cellflow._types import ArrayLike
from cellflow.data._dataloader import DAGEvalAdapter, DAGTrainAdapter
from cellflow.model._base import BaseCellFlow

if TYPE_CHECKING:
    from annbatch import DatasetCollection  # optional dep — only imported for typing

    from dagloader import DAGEvalLoader, SamplerConfig, Scheme  # optional deps — typing only

__all__ = ["CellFlowAnnbatch"]


class CellFlowAnnbatch(BaseCellFlow):
    """CellFlow over the annbatch/dagloader streaming path (cells sourced out-of-core or in-memory).

    Cells are streamed from an :class:`~annbatch.DatasetCollection` (out-of-core) or an in-memory
    ``AnnData`` via :meth:`prepare_data`; validation and prediction read each condition's full cell
    set through :class:`~dagloader.DAGEvalLoader` (no boolean masking). Model setup, training, prediction
    and IO are inherited from :class:`~cellflow.model._base.BaseCellFlow`.
    """

    def __init__(self, solver: Literal["otfm", "genot"] = "otfm"):
        super().__init__(solver)
        self._scheme: Scheme | None = None
        self._split_schemes: dict[str, Scheme] | None = None
        self._split_assignment: pd.DataFrame | None = None
        self._annbatch_sampler_configs: dict[str, SamplerConfig] | None = None
        self._eval_cfg: SamplerConfig | None = None  # target read params for DAGEvalLoader
        self._condition_data: dict[str, np.ndarray] | None = None  # condition embeddings
        self._max_combination_length: int | None = None
        self._condition_fn = None  # leaf -> embedding (set by `prepare_data`)
        self._prep_kwargs: dict[str, Any] | None = None  # covariate spec, reused for validation sources
        self._seed: int = 0
        self._split_eval_loaders: dict[str, DAGEvalLoader] = {}

    def prepare_data(
        self,
        source: "ad.AnnData | DatasetCollection",
        sample_rep: str,
        control_key: str,
        perturbation_covariates: dict[str, Sequence[str]],
        perturbation_covariate_reps: dict[str, str] | None = None,
        sample_covariates: Sequence[str] | None = None,
        sample_covariate_reps: dict[str, str] | None = None,
        split_covariates: Sequence[str] | None = None,
        max_combination_length: int | None = None,
        null_value: float = 0.0,
        rep_dict: Mapping[str, Mapping[str, ArrayLike]] | None = None,
        sampler_config: "SamplerConfig | Mapping[str, SamplerConfig] | None" = None,
        seed: int = 0,
        control_in_memory: bool = True,
        min_cells_per_condition: int = 0,
        split_by: Sequence[str] | None = None,
        split_ratios: Mapping[str, float] | None = None,
        split_force_training_values: Mapping[str, object] | None = None,
        split_random_state: int = 42,
    ) -> None:
        """Prepare the annbatch/dagloader streaming training path (the default, recommended path).

        The covariate arguments (``sample_rep``, ``control_key``, ``perturbation_covariates``,
        ``perturbation_covariate_reps``, ``sample_covariates``, ``sample_covariate_reps``,
        ``split_covariates``, ``max_combination_length``, ``null_value``) mean exactly what they do in
        :meth:`prepare_data`; only the cells differ — streamed from an out-of-core
        :class:`annbatch.DatasetCollection` instead of materialized. Requires the ``annbatch`` extra and
        a model constructed as ``CellFlowAnnbatch()``.

        Parameters
        ----------
        source
            An out-of-core :class:`annbatch.DatasetCollection` to stream cells from (an in-memory
            ``AnnData`` also works — the ``dagloader`` is container-agnostic). Its ``obs`` supplies the
            grouping / condition columns; ``sample_rep`` selects the streamed representation.
        rep_dict
            The covariate embedding tables that ``adata.uns`` would hold in the in-memory path (keys
            match the values of ``perturbation_covariate_reps`` / ``sample_covariate_reps``). Required
            when a covariate group is embedded; may be :obj:`None` when the primary covariate is
            categorical (one-hot encoded).
        sampler_config
            Read parameters for the streamed loader(s), **one per split**: a single
            :class:`dagloader.SamplerConfig` for all splits, or a ``{split_name: SamplerConfig}`` mapping
            covering every split (see :func:`dagloader.resolve_split_configs`). With no split the only
            split is ``"train"``. ``chunk_size > 1`` reads contiguous slices, so every run of each
            category must be at least ``chunk_size`` cells — in-memory sources are grouped automatically,
            an out-of-core :class:`~annbatch.DatasetCollection` must be built grouped
            (``add_adatas(..., groupby=[...])``) or a clear error is raised.
        seed
            Reproducibility seed for the ``dagloader`` per-node RNG streams.
        min_cells_per_condition
            Drop (zero-weight) any perturbed condition with fewer than this many *total* cells — both a
            scientific filter on untrainable tiny conditions and the lever that unblocks ``chunk_size > 1``:
            a dropped (zero-weight) condition is exempt from annbatch's run-length rule, so its short run no
            longer blocks chunked reads. Applied to the training source and to validation sources built from
            the same spec. The default ``0`` drops nothing (behavior unchanged). Note this counts *total*
            cells per condition, so it only unblocks ``chunk_size > 1`` when the kept conditions' per-plate
            *runs* are also ``>= chunk_size``.
        split_by
            If given, split the prepared ``Scheme``'s target combinations into train/val/test in the
            same call (delegates to :meth:`split_annbatch_data`). Columns whose unique combinations are
            held out (⊆ the target columns). If :obj:`None`, no split is made (the model would train on
            all combinations).
        split_ratios
            ``{split_name: fraction}`` summing to 1.0 for the split. Defaults to
            ``{"train": 0.6, "val": 0.2, "test": 0.2}``. Only used when ``split_by`` is given.
        split_force_training_values
            ``{column: value}`` (keys ⊆ ``split_by``) forced into the training split. Only used when
            ``split_by`` is given.
        split_random_state
            Seed for the split's combination shuffle. Only used when ``split_by`` is given.

        Returns
        -------
        :obj:`None`, and sets up the streaming training data used by :meth:`train`.

        Notes
        -----
        The ``"train"`` split feeds :meth:`train`; when a split is made, the ``val`` / ``test`` splits are
        read via :class:`~dagloader.DAGEvalLoader` (see :attr:`split_eval_loaders`), not streamed.
        """
        from cellflow.data._annbatch import build_annbatch_training
        from dagloader import DAGLoader, resolve_split_configs

        # Build the Scheme + condition_fn + condition embeddings from the covariate spec (obs only).
        built = build_annbatch_training(
            source,
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
            split_covariates=split_covariates,
            max_combination_length=max_combination_length,
            null_value=null_value,
            rep_dict=rep_dict,
            seed=seed,
            control_in_memory=control_in_memory,
            min_cells_per_condition=min_cells_per_condition,
        )
        self._scheme = built.scheme
        self._dm = built.data_manager
        self._condition_data = built.condition_data
        self._data_dim = built.data_dim
        self._max_combination_length = built.max_combination_length
        condition_fn = built.condition_fn
        # kept so `prepare_validation_data` / the auto-wired split loaders can build DAGEvalLoaders.
        self._condition_fn = condition_fn
        self._seed = seed
        self._prep_kwargs = {
            "sample_rep": sample_rep,
            "control_key": control_key,
            "perturbation_covariates": perturbation_covariates,
            "perturbation_covariate_reps": perturbation_covariate_reps,
            "sample_covariates": sample_covariates,
            "sample_covariate_reps": sample_covariate_reps,
            "split_covariates": split_covariates,
            "max_combination_length": max_combination_length,
            "null_value": null_value,
            "rep_dict": rep_dict,
            "seed": seed,
            "control_in_memory": control_in_memory,
            "min_cells_per_condition": min_cells_per_condition,
        }

        # Splitting step — kept in `prepare_data` so preparing and splitting are one call.
        if split_by is not None:
            self._split_assignment = self.split_annbatch_data(
                split_by=split_by,
                ratios=split_ratios,
                force_training_values=split_force_training_values,
                random_state=split_random_state,
            )

        # One `SamplerConfig` per split (a single spec ⇒ all splits; a per-split dict ⇒ all specified).
        # The splits are the split schemes when a split was made, else the single ``"train"`` scheme.
        schemes = self._split_schemes if self._split_schemes is not None else {"train": self._scheme}
        if sampler_config is None:
            raise ValueError("`sampler_config` is required: give a SamplerConfig or a {split: SamplerConfig} mapping.")
        self._annbatch_sampler_configs = resolve_split_configs(sampler_config, list(schemes))
        self._eval_cfg = self._annbatch_sampler_configs["train"]  # target-batch read params for eval loaders

        # No cellflow-side run-length pre-check: annbatch enforces the `chunk_size > 1` run-length rule
        # itself when building each node's sampler (raising a clear, class-level error), so a redundant
        # check here would only duplicate that logic. `min_cells_per_condition` zero-weights sub-threshold
        # conditions (annbatch's ClassSampler skips them), and an in-memory control node samples at
        # chunk_size=1 (see DAGLoader) — so only the streamed perturbed layout has to satisfy the rule.

        # Only the "train" split is streamed (feeds `train()`); val/test are read via DAGEvalLoader
        # below, so we don't build unused per-split streaming loaders.
        self._dataloader = DAGTrainAdapter(
            DAGLoader(schemes["train"], self._annbatch_sampler_configs["train"], condition_fn=condition_fn)
        )

        # Auto-wire the val/test split combinations as evaluation sources over the same cells: a
        # `DAGEvalLoader` reads each held-out condition's full cell set + matched controls. The "val"
        # split (if any) feeds training-time metrics; every non-train split is kept on
        # `split_eval_loaders` for post-hoc evaluation. Override with `prepare_validation_data(...)`.
        from dagloader import DAGEvalLoader

        self._split_eval_loaders = {}
        if self._split_schemes is not None:
            for split_name, sch in self._split_schemes.items():
                if split_name == "train":
                    continue
                self._split_eval_loaders[split_name] = DAGEvalLoader(sch, self._eval_cfg, condition_fn, seed=seed)
            if "val" in self._split_eval_loaders:
                self._validation_data["val"] = DAGEvalAdapter(self._split_eval_loaders["val"])

    def split_annbatch_data(
        self,
        *,
        split_by: Sequence[str],
        ratios: Mapping[str, float] | None = None,
        force_training_values: Mapping[str, object] | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Partition the prepared annbatch ``Scheme``'s target combinations into named splits.

        Call after :meth:`prepare_data`. Splits hold out whole *combinations* of ``split_by``
        (a subset of the perturbation / split-covariate columns), not cells; controls are carried through
        every split. See :func:`dagloader.split_scheme` for the mechanics.

        Parameters
        ----------
        split_by
            Columns whose unique combinations are partitioned across splits (⊆ the scheme's target
            columns).
        ratios
            ``{split_name: fraction}`` summing to 1.0. Defaults to
            ``{"train": 0.6, "val": 0.2, "test": 0.2}``. The first split is the training split.
        force_training_values
            ``{column: value}`` (keys ⊆ ``split_by``): any combination matching a value is forced into
            the training (first) split.
        random_state
            Seed for the combination shuffle.

        Returns
        -------
        A :class:`~pandas.DataFrame` of the target combinations and their assigned split. Also stores
        the per-split schemes on the model.
        """
        from dagloader import split_assignment, split_scheme

        if self._scheme is None:
            raise ValueError(
                "No annbatch `Scheme` to split. Call `prepare_data(...)` first "
                "(the out-of-core streaming path)."
            )
        self._split_schemes = split_scheme(
            self._scheme,
            split_by=split_by,
            ratios=ratios,
            force_training_values=force_training_values,
            random_state=random_state,
        )
        # `prepare_data` wires the "train" split into the streaming loader and the val/test splits into
        # `DAGEvalLoader`s (`split_eval_loaders`); here we only produce the schemes + table.
        return split_assignment(self._split_schemes)

    def prepare_validation_data(
        self,
        source: "ad.AnnData | DatasetCollection",
        name: str,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Register a validation set read via :class:`~dagloader.DAGEvalLoader` (no boolean masking).

        The ``source`` (an :class:`~anndata.AnnData` or an :class:`~annbatch.DatasetCollection`) is
        grouped by the same covariate spec passed to :meth:`prepare_data`; each of its conditions is
        read in full (all its cells + all matched controls) at validation time. Overrides any same-named
        auto-wired split loader. Preserves the legacy per-condition metric semantics.

        Parameters
        ----------
        source
            The held-out validation source (out-of-core or in-memory).
        name
            Key under which the validation set is stored in :attr:`validation_data`.
        n_conditions_on_log_iteration, n_conditions_on_train_end
            Conditions to sample per validation step; :obj:`None` uses all.
        predict_kwargs
            Keyword arguments for the solver's ``predict`` used during validation.
        """
        if self._scheme is None or self._prep_kwargs is None:
            raise ValueError("Set up training first via `prepare_data(...)` before preparing validation data.")

        from cellflow.data._annbatch import build_annbatch_training
        from dagloader import DAGEvalLoader

        built = build_annbatch_training(source, **self._prep_kwargs)
        eval_loader = DAGEvalLoader(built.scheme, self._eval_cfg, built.condition_fn, seed=self._seed)
        self._validation_data[name] = DAGEvalAdapter(
            eval_loader,
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        predict_kwargs = predict_kwargs or {}
        if len(self._validation_data.get("predict_kwargs", {})) > 0 and len(predict_kwargs) > 0:
            self._validation_data["predict_kwargs"].update(predict_kwargs)
            predict_kwargs = self._validation_data["predict_kwargs"]
        self._validation_data["predict_kwargs"] = predict_kwargs

    @property
    def split_eval_loaders(self) -> dict[str, "DAGEvalLoader"]:
        """Per-split :class:`~dagloader.DAGEvalLoader` objects for non-train splits (e.g. ``val``, ``test``)."""
        return self._split_eval_loaders

    # ── path-specific hook implementations ───────────────────────────────────────────────────────
    def _encoder_conditions(self) -> tuple[dict[str, np.ndarray], int]:
        if self._condition_data is None or self._max_combination_length is None:
            raise ValueError("Data not initialized. Please call `prepare_data(...)` first.")
        return self._condition_data, self._max_combination_length

    def _bind_train_dataloader(self, batch_size: int, out_of_core_dataloading: bool) -> None:
        # the streaming `_dataloader` (DAGTrainAdapter) was already set in `prepare_data`.
        return None

    def _build_validation_loaders(self) -> dict[str, Any]:
        return {k: v for k, v in self.validation_data.items() if k != "predict_kwargs"}
