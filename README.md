<img src="docs/_static/images/cellflow_dark.png" width="500" alt="CellFlow">

[![PyPI](https://img.shields.io/pypi/v/cellflow-tools.svg)](https://pypi.org/project/cellflow-tools/)
[![Downloads](https://static.pepy.tech/badge/cellflow-tools)](https://pepy.tech/project/cellflow-tools)
[![CI](https://img.shields.io/github/actions/workflow/status/theislab/cellflow/test.yaml?branch=main)](https://github.com/theislab/cellflow/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/theislab/CellFlow/main.svg)](https://results.pre-commit.ci/latest/github/theislab/CellFlow/main)
[![Codecov](https://codecov.io/gh/theislab/cellflow/branch/main/graph/badge.svg?token=Rgtm5Tsblo)](https://codecov.io/gh/theislab/cellflow)
[![Docs](https://img.shields.io/readthedocs/cellflow)](https://cellflow.readthedocs.io/en/latest/)

CellFlow - Modeling Complex Perturbations with Flow Matching 
============================================================

CellFlow is a framework for predicting single-cell phenotypes induced by complex perturbations. It is quite flexible and enables researchers to systematically explore how cells respond to a wide range of experimental interventions, including drug treatments, genetic modifications, cytokine stimulation, morphogen pathway modulation or even entire organoid protocols.

Check out the [preprint](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.abstract) for details!

## Example Applications

- Modeling the effect of single and combinatorial drug treatments
- Predicting the phenotypic response to genetic perturbations
- Modeling the development of perturbed organisms
- Cell fate engineering
- Optimizing protocols for growing organoids
- ... and more; check out the [documentation](https://cellflow.readthedocs.io) for more information.


Installation
------------
Install **CellFlow** by running::

    pip install cellflow-tools


In order to install **CellFlow** in editable mode, run::

    git clone https://github.com/theislab/cellflow
    cd cellflow
    pip install -e .

For further instructions how to install jax, please refer to https://github.com/google/jax.
