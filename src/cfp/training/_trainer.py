from collections.abc import Sequence
from typing import Any, Literal

import jax
from flax.jax_utils import prefetch_to_device
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from cfp.data._dataloader import TrainSampler, ValidationSampler
from cfp.solvers import _genot, _otfm
from cfp.training._callbacks import BaseCallback, CallbackRunner
import collections
import itertools
import warnings
from collections.abc import Iterable  # pylint: disable=g-importing-member

import jax
import jax.numpy as jnp
import numpy as np
from jax import core, lax

class IterativeSampler:
    def __init__(self, dataloader, rng, num_iterations):
        self.dataloader = dataloader
        self.rng = rng
        self.num_iterations = num_iterations


    def __iter__(self):
        for _ in range(self.num_iterations):
            self.rng, rng_data = jax.random.split(self.rng, 2)
            batch = self.dataloader.sample(rng_data)
            yield batch



def prefetch_to_device(iterator, size, devices=None):
    queue = collections.deque()

    def _prefetch(xs):
        return jax.device_put(xs, devices)

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)

class CellFlowTrainer:
    """Trainer for the OTFM/GENOT solver with a conditional velocity field.

    Parameters
    ----------
        dataloader
            Data sampler.
        solver
            OTFM/GENOT solver with a conditional velocity field.
        seed
            Random seed for subsampling validation data.

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        solver: _otfm.OTFlowMatching | _genot.GENOT,
        seed: int = 0,
    ):
        if not isinstance(solver, (_otfm.OTFlowMatching | _genot.GENOT)):
            raise NotImplementedError(
                f"Solver must be an instance of OTFlowMatching or GENOT, got {type(solver)}"
            )

        self.solver = solver
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

        valid_pred_data: dict[str, dict[str, ArrayLike]] = {}
        valid_true_data: dict[str, dict[str, ArrayLike]] = {}
        for val_key, vdl in val_data.items():
            batch = vdl.sample(mode=mode)
            src = batch["source"]
            condition = batch.get("condition", None)
            true_tgt = batch["target"]
            valid_pred_data[val_key] = jax.tree.map(self.solver.predict, src, condition)
            valid_true_data[val_key] = true_tgt

        return valid_true_data, valid_pred_data

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: TrainSampler,
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
        rng = jax.random.PRNGKey(0)

        # Initiate callbacks
        valid_loaders = valid_loaders or {}
        crun = CallbackRunner(
            callbacks=callbacks,
        )
        crun.on_train_begin()

        pbar = tqdm(range(num_iterations))

        rng, rng_data, rng_step_fn = jax.random.split(rng, 3)

        iter_sample = IterativeSampler(dataloader=dataloader, rng=rng_data, num_iterations=num_iterations)


        for it, batch in zip(pbar, prefetch_to_device(iter_sample, 3)):
            rng, rng_step_fn = jax.random.split(rng, 2)
            loss = self.solver.step_fn(rng_step_fn, batch)
            self.training_logs["loss"].append(float(loss))

            if ((it - 1) % valid_freq == 0) and (it > 1):
                # Get predictions from validation data
                valid_true_data, valid_pred_data = self._validation_step(
                    valid_loaders, mode="on_log_iteration"
                )

                metrics = crun.on_log_iteration(valid_true_data, valid_pred_data)
                self._update_logs(metrics)

                # Update progress bar
                mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                postfix_dict = {
                    metric: round(self.training_logs[metric][-1], 3)
                    for metric in monitor_metrics
                }
                postfix_dict["loss"] = round(mean_loss, 3)
                pbar.set_postfix(postfix_dict)
        if num_iterations > 0:
            valid_true_data, valid_pred_data = self._validation_step(
                valid_loaders, mode="on_train_end"
            )
            metrics = crun.on_train_end(valid_true_data, valid_pred_data)
            self._update_logs(metrics)

        self.solver.is_trained = True
        return self.solver
