import abc
from collections.abc import Sequence
from typing import Any, Literal

import jax.tree as jt
import numpy as np
from numpy.typing import ArrayLike

from cfp.data.data import ValidationData
from cfp.metrics.metrics import (
    compute_e_distance,
    compute_r_squared,
    compute_scalar_mmd,
    compute_sinkhorn_div,
)
from cfp.networks import ConditionalVelocityField


class BaseCallback(abc.ABC):

    @abc.abstractmethod
    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def on_log_iteration(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def on_train_end(self, *args: Any, **kwargs: Any) -> Any:
        pass


class LoggingCallback(BaseCallback, abc.ABC):

    @abc.abstractmethod
    def on_train_begin(self) -> Any:
        pass

    @abc.abstractmethod
    def on_log_iteration(self, dict_to_log: dict[str, Any]) -> Any:
        pass

    @abc.abstractmethod
    def on_train_end(self, dict_to_log: dict[str, Any]) -> Any:
        pass


class ComputationCallback(BaseCallback, abc.ABC):

    @abc.abstractmethod
    def on_train_begin(self) -> Any:
        pass

    @abc.abstractmethod
    def on_log_iteration(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        pass

    @abc.abstractmethod
    def on_train_end(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        pass


metric_to_func = {
    "r_squared": compute_r_squared,
    "mmd": compute_scalar_mmd,
    "sinkhorn_div": compute_sinkhorn_div,
    "e_distance": compute_e_distance,
}


class ComputeMetrics(ComputationCallback):
    def __init__(
        self,
        metrics: list[Literal["r_squared", "mmd", "sinkhorn_div", "e_distance"]],
        metric_aggregation: list[Literal["mean", "median"]] = "mean",
    ):
        self.metrics = metrics
        self.metric_aggregation = metric_aggregation
        self._aggregation_func = (
            np.median if metric_aggregation == "median" else np.mean
        )
        for metric in metrics:
            if metric not in metric_to_func:
                raise ValueError(
                    f"Metric {metric} not supported. Supported metrics are {list(metric_to_func.keys())}"
                )

    def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def on_log_iteration(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
        training_data: dict[str, ArrayLike],
    ) -> dict[str, float]:
        metrics = {}
        for metric in self.metrics:
            for k in validation_data.keys():
                result = jt.flatten(
                    jt.map(
                        metric_to_func[metric],
                        validation_data[k].tgt_data,
                        predicted_data[k],
                    )
                )[0]
                # TODO: support multiple aggregation functions
                metrics[f"{k}_{metric}"] = self._aggregation_func(result)
            result = metric_to_func[metric](
                training_data["tgt_data"], training_data["pred_data"]
            )
            metrics[f"train_{metric}"] = self._aggregation_func(result)

        return metrics

    def on_train_end(
        self,
        validation_data: dict[str, ValidationData],
        predicted_data: dict[str, dict[str, ArrayLike]],
        training_data: dict[str, ArrayLike],
    ) -> dict[str, float]:
        return self.on_log_iteration(validation_data, predicted_data)


class WandbLogger(LoggingCallback):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is not installed, please install it via `pip install wandb`"
        )
    try:
        import omegaconf
    except ImportError:
        raise ImportError(
            "omegaconf is not installed, please install it via `pip install omegaconf`"
        )

    def __init__(
        self,
        project: str,
        out_dir: str,
        config: omegaconf.OmegaConf | dict[str, Any],
        **kwargs,
    ):
        self.project = project
        self.out_dir = out_dir
        self.config = config
        self.kwargs = kwargs

    def on_train_begin(self) -> Any:
        if isinstance(self.config, dict):
            config = omegaconf.OmegaConf.create(self.config)
        wandb.login()
        wandb.init(
            project=wandb_project,
            config=omegaconf.OmegaConf.to_container(config, resolve=True),
            dir=out_dir,
            settings=wandb.Settings(
                start_method=self.kwargs.pop("start_method", "thread")
            ),
        )

    def on_log_iteration(
        self,
        dict_to_log: dict[str, float],
        **_: Any,
    ) -> Any:
        wandb.log(dict_to_log)

    def on_train_end(self, dict_to_log: dict[str, float]) -> Any:
        wandb.log(dict_to_log)


class CallbackRunner:
    """Runs a set of computational and logging callbacks in the CellFlowTrainer

    Args:
        computation_callbacks: List of computation callbacks
        logging_callbacks: List of logging callbacks
        data: Validation data to use for computing metrics

    Returns
    -------
        None
    """

    def __init__(
        self,
        callbacks: list[ComputationCallback],
        data: dict[str, ValidationData],
    ) -> None:

        self.validation_data = data
        self.computation_callbacks = [
            c for c in callbacks if isinstance(c, ComputationCallback)
        ]
        self.logging_callbacks = [
            c for c in callbacks if isinstance(c, LoggingCallback)
        ]

        if len(self.computation_callbacks) == 0 & len(self.logging_callbacks) != 0:
            raise ValueError(
                "No computation callbacks defined to compute metrics to log"
            )

    def on_train_begin(self) -> Any:

        for callback in self.computation_callbacks:
            callback.on_train_begin()

        for callback in self.logging_callbacks:
            callback.on_train_begin()

    def on_log_iteration(self, train_data, pred_data) -> dict[str, Any]:
        dict_to_log: dict[str, Any] = {}

        for callback in self.computation_callbacks:
            results = callback.on_log_iteration(
                self.validation_data, pred_data, train_data
            )
            dict_to_log.update(results)

        for callback in self.logging_callbacks:
            callback.on_log_iteration(dict_to_log)

        return dict_to_log

    def on_train_end(self, train_data, pred_data) -> dict[str, Any]:
        dict_to_log: dict[str, Any] = {}

        for callback in self.computation_callbacks:
            results = callback.on_log_iteration(
                self.validation_data, pred_data, train_data
            )
            dict_to_log.update(results)

        for callback in self.logging_callbacks:
            callback.on_log_iteration(dict_to_log)

        return dict_to_log
