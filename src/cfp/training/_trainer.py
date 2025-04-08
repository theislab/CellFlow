from collections.abc import Sequence
from typing import Any, Literal

import jax
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
import threading
import queue
import jax
from jax import core, lax
import time  # Add this import at the top of the file


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


def prefetch_to_device(data_iter, prefetch_size=2, num_threads=4):
    """Prefetch data to device using multiple threads with timing information.

    Parameters
    ----------
    data_iter : iterator
        Data iterator.
    prefetch_size : int
        Size of the prefetch queue.
    num_threads : int
        Number of producer threads to use.
    """
    input_queue = queue.Queue()  # Queue for raw batches
    output_queue = queue.Queue(maxsize=prefetch_size)  # Queue for processed batches

    # Put all batches into the input queue
    def feeder():
        for batch in data_iter:
            input_queue.put(batch)

        # Signal the end for each worker
        for _ in range(num_threads):
            input_queue.put(None)

    # Start the feeder thread
    threading.Thread(target=feeder, daemon=True, name="feeder").start()

    # Producer function that moves data to device
    def producer():
        while True:
            batch = input_queue.get()
            if batch is None:
                output_queue.put(None)
                break

            # start_time = time.time()
            device_batch = jax.device_put(batch, jax.devices()[0])
            jax.block_until_ready(device_batch)

            # elapsed = time.time() - start_time

            # print(f"Thread {threading.current_thread().name}: Moving batch to device took {elapsed:.4f} seconds")
            output_queue.put(device_batch)

    # Start multiple producer threads
    for i in range(num_threads):
        t = threading.Thread(target=producer, daemon=True, name=f"producer-{i}")
        t.start()

    # Count termination signals received
    end_signals_received = 0

    # Yield prefetched batches
    while end_signals_received < num_threads:
        batch = output_queue.get()
        if batch is None:
            end_signals_received += 1
        else:
            yield batch


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
            raise NotImplementedError(f"Solver must be an instance of OTFlowMatching or GENOT, got {type(solver)}")

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

        rng, rng_data, rng_gpu = jax.random.split(rng, 3)
        rng_gpu = jax.device_put(rng_gpu)

        iter_sample = IterativeSampler(dataloader=dataloader, rng=rng_data, num_iterations=num_iterations)

        self.solver.vf_state = jax.device_put(self.solver.vf_state, jax.devices()[0])

        for it, batch in zip(pbar, prefetch_to_device(iter_sample, 16, 4)):
            rng_gpu, rng_step_fn = jax.random.split(rng_gpu, 2)
            loss = self.solver.step_fn(rng_step_fn, batch)
            self.training_logs["loss"].append(float(loss))
            # print(f"Iteration {it}: Loss: {loss:.4f}, Time: {time.time() - ts:.4f} seconds")
            if ((it - 1) % valid_freq == 0) and (it > 1):
                # Get predictions from validation data
                valid_true_data, valid_pred_data = self._validation_step(valid_loaders, mode="on_log_iteration")

                metrics = crun.on_log_iteration(valid_true_data, valid_pred_data)
                self._update_logs(metrics)

                # Update progress bar
                mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                postfix_dict = {metric: round(self.training_logs[metric][-1], 3) for metric in monitor_metrics}
                postfix_dict["loss"] = round(mean_loss, 3)
                pbar.set_postfix(postfix_dict)
        if num_iterations > 0:
            valid_true_data, valid_pred_data = self._validation_step(valid_loaders, mode="on_train_end")
            metrics = crun.on_train_end(valid_true_data, valid_pred_data)
            self._update_logs(metrics)

        self.solver.is_trained = True
        return self.solver
