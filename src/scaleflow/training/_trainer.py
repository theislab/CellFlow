from collections.abc import Sequence
from typing import Any, Literal

import jax
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from scaleflow.data import SamplerABC
from scaleflow.solvers import _eqm, _genot, _otfm
from scaleflow.training._callbacks import BaseCallback, CallbackRunner


class CellFlowTrainer:
    """Trainer for the OTFM/GENOT/EqM solver with a conditional velocity field.

    Parameters
    ----------
        dataloader
            Data sampler.
        solver
            :class:`~scaleflow.solvers._otfm.OTFlowMatching`,
            :class:`~scaleflow.solvers._genot.GENOT`, or
            :class:`~scaleflow.solvers._eqm.EquilibriumMatching` solver with a conditional velocity field.
        predict_kwargs
            Keyword arguments for the prediction functions
            :func:`scaleflow.solvers._otfm.OTFlowMatching.predict`,
            :func:`scaleflow.solvers._genot.GENOT.predict`, or
            :func:`scaleflow.solvers._eqm.EquilibriumMatching.predict` used during validation.
        seed
            Random seed for subsampling validation data.

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        solver: _otfm.OTFlowMatching | _genot.GENOT | _eqm.EquilibriumMatching,
        predict_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        if not isinstance(solver, (_otfm.OTFlowMatching | _genot.GENOT | _eqm.EquilibriumMatching)):
            raise NotImplementedError(
                f"Solver must be an instance of OTFlowMatching, GENOT, or EquilibriumMatching, got {type(solver)}"
            )

        self.solver = solver
        self.predict_kwargs = predict_kwargs or {}
        self.rng_subsampling = np.random.default_rng(seed)
        self.training_logs: dict[str, Any] = {}

    def _validation_step(
        self,
        val_data: dict[str, SamplerABC],
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

        # Add progress bar for validation
        val_pbar = tqdm(val_data.items(), desc="Validation", leave=False)
        for val_key, vdl in val_pbar:
            batch = vdl.sample(mode=mode) # TODO: remove mode
            src = batch["source"]
            condition = batch.get("condition", None)
            true_tgt = batch["target"]
            valid_source_data[val_key] = src
            valid_pred_data[val_key] = self.solver.predict(src, condition=condition, **self.predict_kwargs)
            valid_true_data[val_key] = true_tgt

            # Update progress bar description with current validation set
            val_pbar.set_description(f"Validation ({val_key})")

        return valid_source_data, valid_true_data, valid_pred_data

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: SamplerABC,
        num_iterations: int,
        valid_freq: int,
        valid_loaders: dict[str, SamplerABC] | None = None,
        monitor_metrics: Sequence[str] = [],
        callbacks: Sequence[BaseCallback] = [],
    ) -> _otfm.OTFlowMatching | _genot.GENOT | _eqm.EquilibriumMatching:
        """Trains the model.

        Parameters
        ----------
            dataloader
                Dataloader used. The dataloader is responsible for returning batches
                with appropriate 'task' field ('gex' or 'functional').
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
        self.training_logs = {"loss": [], "loss_gex": [], "loss_functional": []}
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
        sampler.init_sampler(rng_np)
        for it in pbar:
            rng_jax, rng_step_fn = jax.random.split(rng_jax, 2)

            # Sample batch (dataloader controls which task)
            batch = sampler.sample(rng_np)
            loss = self.solver.step_fn(rng_step_fn, batch)

            # Track losses
            task = batch.get("task", "gex")
            self.training_logs["loss"].append(float(loss))
            self.training_logs[f"loss_{task}"].append(float(loss))

            if ((it - 1) % valid_freq == 0) and (it > 1):
                # Get predictions from validation data
                valid_source_data, valid_true_data, valid_pred_data = self._validation_step(
                    valid_loaders, mode="on_log_iteration"
                )

                # Calculate mean losses
                mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                additional_metrics = {"train_loss": mean_loss}

                # Add task-specific losses if available
                if self.training_logs["loss_gex"]:
                    mean_loss_gex = np.mean([l for l in self.training_logs["loss_gex"][-valid_freq:] if l is not None])
                    additional_metrics["train_loss_gex"] = mean_loss_gex
                if self.training_logs["loss_functional"]:
                    mean_loss_func = np.mean([l for l in self.training_logs["loss_functional"][-valid_freq:] if l is not None])
                    additional_metrics["train_loss_functional"] = mean_loss_func

                # Run callbacks with loss as additional metric
                metrics = crun.on_log_iteration(
                    valid_source_data,
                    valid_true_data,
                    valid_pred_data,
                    self.solver,
                    additional_metrics=additional_metrics,
                    iteration=it,
                )
                self._update_logs(metrics)
                # Update progress bar
                postfix_dict = {metric: round(self.training_logs[metric][-1], 3) for metric in monitor_metrics}
                postfix_dict["train_loss"] = round(mean_loss, 3)
                if "train_loss_gex" in additional_metrics:
                    postfix_dict["loss_gex"] = round(additional_metrics["train_loss_gex"], 3)
                if "train_loss_functional" in additional_metrics:
                    postfix_dict["loss_func"] = round(additional_metrics["train_loss_functional"], 3)
                pbar.set_postfix(postfix_dict)

        if num_iterations > 0:
            valid_source_data, valid_true_data, valid_pred_data = self._validation_step(
                valid_loaders, mode="on_train_end"
            )
            metrics = crun.on_train_end(valid_source_data, valid_true_data, valid_pred_data, self.solver)
            self._update_logs(metrics)

        self.solver.is_trained = True
        return self.solver
