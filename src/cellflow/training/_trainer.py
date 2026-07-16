from collections.abc import Sequence
from typing import Any, Literal

import jax
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from cellflow.data._legacy import OOCTrainSampler, TrainSampler, ValidationSampler
from cellflow.solvers import _genot, _otfm
from cellflow.training._callbacks import BaseCallback, CallbackRunner


class CellFlowTrainer:
    """Trainer for the OTFM/GENOT solver with a conditional velocity field.

    Parameters
    ----------
        dataloader
            Data sampler.
        solver
            :class:`~cellflow.solvers._otfm.OTFlowMatching` or
            :class:`~cellflow.solvers._genot.GENOT` solver with a conditional velocity field.
        predict_kwargs
            Keyword arguments for the prediction functions
            :func:`cellflow.solvers._otfm.OTFlowMatching.predict` or
            :func:`cellflow.solvers._genot.GENOT.predict` used during validation.
        seed
            Random seed for subsampling validation data.

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        solver: _otfm.OTFlowMatching | _genot.GENOT,
        predict_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        if not isinstance(solver, (_otfm.OTFlowMatching | _genot.GENOT)):
            raise NotImplementedError(f"Solver must be an instance of OTFlowMatching or GENOT, got {type(solver)}")

        self.solver = solver
        self.predict_kwargs = predict_kwargs or {}
        self.rng_subsampling = np.random.default_rng(seed)
        self.training_logs: dict[str, Any] = {}

    def _validation_step(
        self,
        val_data: dict[str, ValidationSampler],
        mode: Literal["on_log_iteration", "on_train_end"] = "on_log_iteration",
    ) -> tuple[
        dict[str, dict[str, ArrayLike]],
        dict[str, dict[str, ArrayLike]],
    ]:
        """Compute predictions for validation data."""
        # TODO: Sample fixed number of conditions to validate on

        valid_source_data: dict[str, dict[str, ArrayLike]] = {}
        valid_pred_data: dict[str, dict[str, ArrayLike]] = {}
        valid_true_data: dict[str, dict[str, ArrayLike]] = {}
        for val_key, vdl in val_data.items():
            batch = vdl.sample(mode=mode)
            src = batch["source"]
            condition = batch.get("condition", None)
            true_tgt = batch["target"]
            valid_source_data[val_key] = src
            valid_pred_data[val_key] = self.solver.predict(src, condition=condition, **self.predict_kwargs)
            valid_true_data[val_key] = true_tgt

        return valid_source_data, valid_true_data, valid_pred_data

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: TrainSampler | OOCTrainSampler,
        num_iterations: int,
        valid_freq: int,
        valid_loaders: dict[str, ValidationSampler] | None = None,
        monitor_metrics: Sequence[str] = [],
        callbacks: Sequence[BaseCallback] = [],
    ) -> _otfm.OTFlowMatching | _genot.GENOT:
        """Trains the model.

        Parameters
        ----------
            dataloader
                Dataloader used.
            num_iterations
                Number of iterations to train the model.
            valid_freq
                Frequency of validation.
            valid_loaders
                Valid loaders.
            callbacks
                Callback functions.
            monitor_metrics
                Metrics to monitor.

        Returns
        -------
            The trained model.
        """
        self.training_logs = {"loss": []}
        rng_jax = jax.random.PRNGKey(0)
        rng_np = np.random.default_rng(0)

        # Initiate callbacks
        valid_loaders = valid_loaders or {}
        crun = CallbackRunner(
            callbacks=callbacks,
        )
        crun.on_train_begin()

        pbar = tqdm(range(num_iterations))
        sampler = dataloader
        if isinstance(dataloader, OOCTrainSampler):
            dataloader.set_sampler(num_iterations=num_iterations)

        # Keep per-step losses on-device and materialize them in bulk. Calling ``float(loss)``
        # every iteration forces a device->host sync that serializes host-side batch sampling
        # behind GPU compute (GPU bubbles / underutilization). Buffering + a periodic flush
        # removes ~all per-step syncs while bounding the async look-ahead so device memory
        # cannot run away, and keeps ``training_logs["loss"]`` as plain floats (unchanged API).
        loss_buf: list[Any] = []
        flush_every = min(valid_freq, 50) if valid_freq > 0 else 50

        def _flush_losses() -> None:
            if loss_buf:
                self.training_logs["loss"].extend(np.asarray(jax.device_get(loss_buf)).ravel().tolist())
                loss_buf.clear()

        for it in pbar:
            rng_jax, rng_step_fn = jax.random.split(rng_jax, 2)
            batch = sampler.sample(rng_np)
            loss = self.solver.step_fn(rng_step_fn, batch)
            loss_buf.append(loss)  # on-device; no host sync on the hot path
            if len(loss_buf) >= flush_every:
                _flush_losses()

            if ((it - 1) % valid_freq == 0) and (it > 1):
                _flush_losses()
                # Get predictions from validation data
                valid_source_data, valid_true_data, valid_pred_data = self._validation_step(
                    valid_loaders, mode="on_log_iteration"
                )

                # Run callbacks
                metrics = crun.on_log_iteration(valid_source_data, valid_true_data, valid_pred_data, self.solver)  # type: ignore[arg-type]
                self._update_logs(metrics)

                # Update progress bar
                mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                postfix_dict = {metric: round(self.training_logs[metric][-1], 3) for metric in monitor_metrics}
                postfix_dict["loss"] = round(mean_loss, 3)
                pbar.set_postfix(postfix_dict)

        _flush_losses()
        if num_iterations > 0:
            valid_source_data, valid_true_data, valid_pred_data = self._validation_step(
                valid_loaders, mode="on_train_end"
            )
            metrics = crun.on_train_end(valid_source_data, valid_true_data, valid_pred_data, self.solver)
            self._update_logs(metrics)

        self.solver.is_trained = True
        return self.solver
