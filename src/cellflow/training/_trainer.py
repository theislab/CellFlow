import queue
import threading
import time  # Add this import at the top of the file
from collections.abc import Sequence
from typing import Any, Literal
import jax.numpy as jnp
import jax
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from cellflow.data._dataloader import CpuTrainSampler, ValidationSampler
from cellflow.solvers import _genot, _otfm
from cellflow.training._callbacks import BaseCallback, CallbackRunner


def prefetch_to_device(sampler, num_iterations, prefetch_factor=2, num_workers=4):
    q = queue.Queue(maxsize=prefetch_factor*num_workers)
    sem = threading.Semaphore(num_iterations)
    stop_event = threading.Event()
    def worker(seed):
        rng = np.random.default_rng(seed)
        while not stop_event.is_set() and sem.acquire(blocking=False):
            batch = sampler.sample(rng)
            batch = jax.device_put(batch, jax.devices()[0], donate=True)
            jax.block_until_ready(batch)
            while not stop_event.is_set():
                try:
                    q.put(batch, timeout=1.0)
                    break  # Batch successfully put into the queue; break out of retry loop
                except queue.Full:
                    continue

        return

    # Start multiple worker threads
    ts = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, daemon=True, name=f"worker-{i}", args=(i, ))
        t.start()
        ts.append(t)

    try:
        for i in range(num_iterations):
            # Yield batches from the queue; will block waiting for available batch
            yield q.get()
    finally:
        # When the generator is closed or garbage collected, clean up the worker threads
        stop_event.set()  # Signal all workers to exit
        for t in ts:
            t.join()  # Wait for all worker threads to finish




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
            batch = jax.device_put(batch, jax.devices()[0], donate=True)
            src = batch["source"]
            
            condition = batch.get("condition", None)
            true_tgt = batch["target"]
            print(src.keys())
            print(condition.keys())

            keys = sorted(src.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            batched_src = jnp.concatenate([src[k] for k in keys], axis=0)
            print(batched_src.shape)
                        
            batched_condition = jnp.concatenate(jnp.concatenate([condition[k][ck] for k in keys for ck in condition_keys], axis=-1), axis=0)
            print(batched_condition.shape)

            # def jitted_solver(src, condition_arr, keys, keys2idx):
            #     return self.solver(src, {k: condition_arr[keys2idx[k]] for k in keys})


            # res = []
            # for k in keys:
            #     # src[k] is jnp.ndarray
            #     # condition[k] is a dict[str, jnp.ndarray]
            #     res.append(jitted_solver(src[k], condition[k], keys, keys2idx))
            
            # batched_src = jnp.concatenate([src[k] for k in keys], axis=0)
            # batched_condition = jnp.concatenate([condition[k] for k in keys], axis=0)
            # print(res)
            # valid_true_data[val_key] = true_tgt
            

        return valid_true_data, valid_pred_data

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: CpuTrainSampler,
        num_iterations: int,
        valid_freq: int,
        valid_loaders: dict[str, ValidationSampler] | None = None,
        monitor_metrics: Sequence[str] = [],
        callbacks: Sequence[BaseCallback] = [],
        num_workers:int =4,
        prefetch_factor:int =4,
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

        rng, rng_gpu = jax.random.split(rng, 2)
        rng_gpu = jax.device_put(rng_gpu, jax.devices()[0])

        self.solver.vf_state = jax.device_put(self.solver.vf_state, jax.devices()[0], donate=True)
        jax.block_until_ready(self.solver.vf_state)
        iter_sample = prefetch_to_device(
            num_iterations=num_iterations,
            sampler=dataloader,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers
        )

        for it in pbar:
            rng_gpu, rng_step_fn = jax.random.split(rng_gpu, 2)
            t0 = time.time()
            batch = next(iter_sample)
            # print(f"Time taken for data loading: {time.time() - t0:.4f} seconds")
            t0 = time.time()
            loss = self.solver.step_fn(rng_step_fn, batch)
            self.training_logs["loss"].append(float(loss))

            if ((it - 1) % valid_freq == 0) and (it > 1):
                # Get predictions from validation data
                valid_true_data, valid_pred_data = self._validation_step(valid_loaders, mode="on_log_iteration")

                # Run callbacks
                metrics = crun.on_log_iteration(valid_true_data, valid_pred_data)  # type: ignore[arg-type]
                self._update_logs(metrics)

                # Update progress bar
                mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                postfix_dict = {metric: round(self.training_logs[metric][-1], 3) for metric in monitor_metrics}
                postfix_dict["loss"] = round(mean_loss, 3)
                pbar.set_postfix(postfix_dict)
        print("HERE IT IS DONE")
        if num_iterations > 0:
            valid_true_data, valid_pred_data = self._validation_step(valid_loaders, mode="on_train_end")
            metrics = crun.on_train_end(valid_true_data, valid_pred_data)
            self._update_logs(metrics)
        print("HERE IT IS DONE 2")
        self.solver.is_trained = True
        return self.solver