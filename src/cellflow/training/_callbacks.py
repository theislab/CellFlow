import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal

import anndata as ad
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np

from cellflow._types import ArrayLike
from cellflow.metrics._metrics import (
    compute_e_distance_fast,
    compute_r_squared,
    compute_scalar_mmd,
    compute_sinkhorn_div,
)

__all__ = [
    "BaseCallback",
    "LoggingCallback",
    "ComputationCallback",
    "Metrics",
    "WandbLogger",
    "CallbackRunner",
    "PCADecodedMetrics",
    "VAEDecodedMetrics",
]


metric_to_func: dict[str, Callable[[ArrayLike, ArrayLike], float | ArrayLike]] = {
    "r_squared": compute_r_squared,
    "mmd": compute_scalar_mmd,
    "sinkhorn_div": compute_sinkhorn_div,
    "e_distance": compute_e_distance_fast,
}

agg_fn_to_func: dict[str, Callable[[ArrayLike], float | ArrayLike]] = {
    "mean": lambda x: np.mean(x, axis=0),
    "median": lambda x: np.median(x, axis=0),
}


class BaseCallback(abc.ABC):
    """Base class for callbacks in the :class:`~cellflow.training.CellFlowTrainer`"""

    @abc.abstractmethod
    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        """Called at the beginning of training"""
        pass

    @abc.abstractmethod
    def on_log_iteration(self, *args: Any, **kwargs: Any) -> Any:
        """Called at each validation/log iteration"""
        pass

    @abc.abstractmethod
    def on_train_end(self, *args: Any, **kwargs: Any) -> Any:
        """Called at the end of training"""
        pass


class LoggingCallback(BaseCallback, abc.ABC):
    """Base class for logging callbacks in the :class:`~cellflow.training.CellFlowTrainer`"""

    @abc.abstractmethod
    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate logging"""
        pass

    @abc.abstractmethod
    def on_log_iteration(self, dict_to_log: dict[str, Any]) -> Any:
        """Called at each validation/log iteration to log data

        Parameters
        ----------
        dict_to_log
            Dictionary containing data to log
        """
        pass

    @abc.abstractmethod
    def on_train_end(self, dict_to_log: dict[str, Any]) -> Any:
        """Called at the end of trainging to log data

        Parameters
        ----------
        dict_to_log
            Dictionary containing data to log
        """
        pass


class ComputationCallback(BaseCallback, abc.ABC):
    """Base class for computation callbacks in the :class:`~cellflow.training.CellFlowTrainer`"""

    @abc.abstractmethod
    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate metric computation"""
        pass

    @abc.abstractmethod
    def on_log_iteration(
        self,
        validation_data: dict[str, dict[str, ArrayLike]],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at each validation/log iteration to compute metrics

        Parameters
        ----------
        validation_data
            Validation data in nested dictionary format with same keys as ``predicted_data``
        predicted_data
            Predicted data in nested dictionary format with same keys as ``validation_data``

        Returns
        -------
            Statistics of the validation data and predicted data
        """
        pass

    @abc.abstractmethod
    def on_train_end(
        self,
        validation_data: dict[str, dict[str, ArrayLike]],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at the end of training to compute metrics

        Parameters
        ----------
        validation_data
            Validation data in nested dictionary format with same keys as ``predicted_data``
        predicted_data
            Predicted data in nested dictionary format with same keys as ``validation_data``

        Returns
        -------
            Statistics of the validation data and predicted data
        """
        pass


class Metrics(ComputationCallback):
    """Callback to compute metrics on validation data during training

    Parameters
    ----------
    metrics
        List of metrics to compute
    metric_aggregations
        List of aggregation functions to use for each metric

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        metrics: list[Literal["r_squared", "mmd", "sinkhorn_div", "e_distance"]],
        metric_aggregations: list[Literal["mean", "median"]] = None,
    ):
        self.metrics = metrics
        self.metric_aggregation = ["mean"] if metric_aggregations is None else metric_aggregations
        for metric in metrics:
            # TODO: support custom callables as metrics
            if metric not in metric_to_func:
                raise ValueError(f"Metric {metric} not supported. Supported metrics are {list(metric_to_func.keys())}")

    def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
        """Called at the beginning of training."""
        pass

    def on_log_iteration(
        self,
        validation_data: dict[str, dict[str, ArrayLike]],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at each validation/log iteration to compute metrics

        Parameters
        ----------
        validation_data
            Validation data in nested dictionary format with same keys as ``predicted_data``
        predicted_data
            Predicted data in nested dictionary format with same keys as ``valid_data``
        """
        metrics = {}
        for metric in self.metrics:
            for k in validation_data.keys():
                out = jtu.tree_map(metric_to_func[metric], validation_data[k], predicted_data[k])
                out_flattened = jt.flatten(out)[0]
                for agg_fn in self.metric_aggregation:
                    metrics[f"{k}_{metric}_{agg_fn}"] = agg_fn_to_func[agg_fn](out_flattened)

        return metrics  # type: ignore[return-value]

    def on_train_end(
        self,
        validation_data: dict[str, dict[str, ArrayLike]],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at the end of training to compute metrics

        Parameters
        ----------
        validation_data
            Validation data in nested dictionary format with same keys as ``predicted_data``
        predicted_data
            Predicted data in nested dictionary format with same keys as ``validation_data``
        """
        return self.on_log_iteration(validation_data, predicted_data)


class PCADecodedMetrics(Metrics):
    """Callback to compute metrics on decoded validation data during training

    Parameters
    ----------
    ref_adata
        An :class:`~anndata.AnnData` object with the reference data containing
        ``adata.varm["X_mean"]`` and ``adata.varm["PCs"]``.
    metrics
        List of metrics to compute. Supported metrics are ``"r_squared"``, ``"mmd"``,
        ``"sinkhorn_div"``, and ``"e_distance"``.
    metric_aggregations
        List of aggregation functions to use for each metric. Supported aggregations are ``"mean"``
        and ``"median"``.
    log_prefix
        Prefix to add to the log keys.
    """

    def __init__(
        self,
        ref_adata: ad.AnnData,
        metrics: list[Literal["r_squared", "mmd", "sinkhorn_div", "e_distance"]],
        metric_aggregations: list[Literal["mean", "median"]] = None,
        log_prefix: str = "pca_decoded_",
    ):
        super().__init__(metrics, metric_aggregations)
        self.pcs = ref_adata.varm["PCs"]
        self.means = ref_adata.varm["X_mean"]
        self.reconstruct_data = lambda x: x @ np.transpose(self.pcs) + np.transpose(self.means)
        self.log_prefix = log_prefix

    def on_log_iteration(
        self,
        validation_data: dict[str, dict[str, ArrayLike]],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at each validation/log iteration to reconstruct the data and compute metrics on the reconstruction

        Parameters
        ----------
        validation_data
            Validation data in nested dictionary format with same keys as ``predicted_data``
        predicted_data
            Predicted data in nested dictionary format with same keys as ``validation_data``
        """
        validation_data_decoded = jtu.tree_map(self.reconstruct_data, validation_data)
        predicted_data_decoded = jtu.tree_map(self.reconstruct_data, predicted_data)

        metrics = super().on_log_iteration(validation_data_decoded, predicted_data_decoded)
        metrics = {f"{self.log_prefix}{k}": v for k, v in metrics.items()}
        return metrics


class VAEDecodedMetrics(Metrics):
    """Callback to compute metrics on decoded validation data during training

    Parameters
    ----------
    vae
        A VAE model object with a ``'get_reconstruction'`` method, can be an instance
        of :class:`cellflow.external.CFJaxSCVI`.
    adata
        An :class:`~anndata.AnnData` object in the same format as the ``vae``.
    metrics
        List of metrics to compute. Supported metrics are ``"r_squared"``, ``"mmd"``,
        ``"sinkhorn_div"``, and ``"e_distance"``.
    metric_aggregations
        List of aggregation functions to use for each metric. Supported aggregations are ``"mean"``
        and ``"median"``.
    log_prefix
        Prefix to add to the log keys.
    """

    def __init__(
        self,
        vae: Callable[[ArrayLike], ArrayLike],
        adata: ad.AnnData,
        metrics: list[Literal["r_squared", "mmd", "sinkhorn_div", "e_distance"]],
        metric_aggregations: list[Literal["mean", "median"]] = None,
        log_prefix: str = "vae_decoded_",
    ):
        super().__init__(metrics, metric_aggregations)
        self.vae = vae
        self._adata_obs = adata.obs.copy()
        self._adata_n_vars = adata.n_vars
        self.reconstruct_data = self.vae.get_reconstructed_expression  # type: ignore[attr-defined]
        self.log_prefix = log_prefix

    def on_log_iteration(
        self,
        validation_data: dict[str, dict[str, ArrayLike]],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        """Called at each validation/log iteration to reconstruct the data and compute metrics on the reconstruction

        Parameters
        ----------
        validation_data
            Validation data in nested dictionary format with same keys as ``predicted_data``
        predicted_data
            Predicted data in nested dictionary format with same keys as ``validation_data``
        """
        validation_data_in_anndata = jtu.tree_map(self._create_anndata, validation_data)
        predicted_data_in_anndata = jtu.tree_map(self._create_anndata, predicted_data)

        validation_data_decoded = jtu.tree_map(self.reconstruct_data, validation_data_in_anndata)
        predicted_data_decoded = jtu.tree_map(self.reconstruct_data, predicted_data_in_anndata)

        metrics = super().on_log_iteration(validation_data_decoded, predicted_data_decoded)
        metrics = {f"{self.log_prefix}{k}": v for k, v in metrics.items()}
        return metrics

    def _create_anndata(self, data: ArrayLike) -> ad.AnnData:
        adata = ad.AnnData(
            X=np.empty((len(data), self._adata_n_vars)),
            obs=self._adata_obs[: len(data)],
        )
        adata.obsm["X_scVI"] = data  # TODO: make package constant
        return adata


class WandbLogger(LoggingCallback):
    """Callback to log data to Weights and Biases

    Parameters
    ----------
    project
        The project name in wandb
    out_dir
        The output directory to save the logs
    config
        The configuration to log
    **kwargs
        Additional keyword arguments to pass to :func:`wandb.init`

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        project: str,
        out_dir: str,
        config: dict[str, Any],
        **kwargs,
    ):
        self.project = project
        self.out_dir = out_dir
        self.config = config
        self.kwargs = kwargs

        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb is not installed, please install it via `pip install wandb`") from None
        try:
            import omegaconf

            self.omegaconf = omegaconf
        except ImportError:
            raise ImportError("omegaconf is not installed, please install it via `pip install omegaconf`") from None

    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate WandB logging"""
        if isinstance(self.config, dict):
            config = self.omegaconf.OmegaConf.create(self.config)
        self.wandb.login()
        self.wandb.init(
            project=self.project,
            config=self.omegaconf.OmegaConf.to_container(config, resolve=True),
            dir=self.out_dir,
            settings=self.wandb.Settings(start_method=self.kwargs.pop("start_method", "thread")),
        )

    def on_log_iteration(
        self,
        dict_to_log: dict[str, float],
        **_: Any,
    ) -> Any:
        """Called at each validation/log iteration to log data to WandB"""
        self.wandb.log(dict_to_log)

    def on_train_end(self, dict_to_log: dict[str, float]) -> Any:
        """Called at the end of training to log data to WandB"""
        self.wandb.log(dict_to_log)
        self.wandb.finish()


class CallbackRunner:
    """Runs a set of computational and logging callbacks in the :class:`~cellflow.training.CellFlowTrainer`

    Parameters
    ----------
    callbacks
        List of callbacks to run. Callbacks should be of type
        :class:`~cellflow.training.ComputationCallback` or
        :class:`~cellflow.training.LoggingCallback`

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        callbacks: Sequence[BaseCallback],
    ) -> None:
        self.computation_callbacks: list[ComputationCallback] = [
            c for c in callbacks if isinstance(c, ComputationCallback)
        ]
        self.logging_callbacks: list[LoggingCallback] = [c for c in callbacks if isinstance(c, LoggingCallback)]

        if len(self.computation_callbacks) == 0 & len(self.logging_callbacks) != 0:
            raise ValueError("No computation callbacks defined to compute metrics to log")

    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate callbacks"""
        for callback in self.computation_callbacks:
            callback.on_train_begin()

        for callback in self.logging_callbacks:
            callback.on_train_begin()

    def on_log_iteration(
        self,
        valid_data: dict[str, dict[str, ArrayLike]],
        pred_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, Any]:
        """Called at each validation/log iteration to run callbacks. First computes metrics with computation callbacks and then logs data with logging callbacks.

        Parameters
        ----------
        valid_data
            Validation data in nested dictionary format with same keys as ``pred_data``
        pred_data
            Predicted data in nested dictionary format with same keys as ``valid_data``

        Returns
        -------
            ``dict_to_log``: Dictionary containing data to log
        """
        dict_to_log: dict[str, Any] = {}

        for callback in self.computation_callbacks:
            results = callback.on_log_iteration(valid_data, pred_data)
            dict_to_log.update(results)

        for callback in self.logging_callbacks:
            callback.on_log_iteration(dict_to_log)  # type: ignore[call-arg]

        return dict_to_log

    def on_train_end(self, valid_data, pred_data) -> dict[str, Any]:
        """Called at the end of training to run callbacks. First computes metrics with computation callbacks and then logs data with logging callbacks.

        Parameters
        ----------
        valid_data: dict
            Validation data in nested dictionary format with same keys as ``pred_data``
        pred_data: dict
            Predicted data in nested dictionary format with same keys as ``valid_data``

        Returns
        -------
            ``dict_to_log``: Dictionary containing data to log
        """
        dict_to_log: dict[str, Any] = {}

        for callback in self.computation_callbacks:
            results = callback.on_log_iteration(valid_data, pred_data)
            dict_to_log.update(results)

        for callback in self.logging_callbacks:
            callback.on_log_iteration(dict_to_log)  # type: ignore[call-arg]

        return dict_to_log
