import jax.tree_util as jtu
import numpy as np
import pytest

import scaleflow


class TestMetrics:
    @pytest.mark.parametrize("prefix", ["", "test_"])
    def test_compute_metrics(self, metrics_data, prefix):
        x_test = metrics_data["x_test"]
        y_test = metrics_data["y_test"]

        metrics = jtu.tree_map(scaleflow.metrics.compute_metrics, x_test, y_test)
        mean_metrics = scaleflow.metrics.compute_mean_metrics(metrics, prefix)

        assert "Alvespimycin+Pirarubicin" in metrics.keys()
        assert {"r_squared", "sinkhorn_div_1", "sinkhorn_div_10", "sinkhorn_div_100", "e_distance", "mmd"} == set(
            metrics["Alvespimycin+Pirarubicin"].keys()
        )
        assert {
            prefix + "r_squared",
            prefix + "sinkhorn_div_1",
            prefix + "sinkhorn_div_10",
            prefix + "sinkhorn_div_100",
            prefix + "e_distance",
            prefix + "mmd",
        } == set(mean_metrics.keys())

    @pytest.mark.parametrize("epsilon", [0.1, 1, 10])
    def test_function_output(self, metrics_data, epsilon):
        x_test = metrics_data["x_test"]["Alvespimycin+Pirarubicin"]
        y_test = metrics_data["y_test"]["Alvespimycin+Pirarubicin"]

        r_squared = scaleflow.metrics.compute_r_squared(x_test, y_test)
        sinkhorn_div = scaleflow.metrics.compute_sinkhorn_div(x_test, y_test, epsilon=epsilon)
        e_distance = scaleflow.metrics.compute_e_distance(x_test, y_test)
        e_distance_fast = scaleflow.metrics.compute_e_distance_fast(x_test, y_test)
        scalar_mmd = scaleflow.metrics.compute_scalar_mmd(x_test, y_test)
        mmd_fast = scaleflow.metrics.maximum_mean_discrepancy(x_test, y_test, exact=False)

        assert -1000 <= r_squared <= 1
        assert sinkhorn_div >= 0
        assert e_distance >= 0
        assert e_distance_fast >= 0
        assert scalar_mmd >= 0
        assert mmd_fast >= 0

    @pytest.mark.parametrize("gamma", [0.1, 1, 2])
    def test_fast_metrics(self, metrics_data, gamma):
        x_test = metrics_data["x_test"]["Alvespimycin+Pirarubicin"]
        y_test = metrics_data["y_test"]["Alvespimycin+Pirarubicin"]

        e_distance = scaleflow.metrics.compute_e_distance(x_test, y_test)
        e_distance_fast = scaleflow.metrics.compute_e_distance_fast(x_test, y_test)

        mmd = scaleflow.metrics.maximum_mean_discrepancy(x_test, y_test, gamma, exact=True)
        mmd_fast = scaleflow.metrics.maximum_mean_discrepancy(x_test, y_test, gamma, exact=False)

        assert np.allclose(e_distance, e_distance_fast, rtol=1e-4, atol=1e-4)
        assert np.allclose(mmd, mmd_fast, rtol=1e-4, atol=1e-4)
