# A macrocircuit--macrocircuit framework

[:link: Read the preprint!](https://www.biorxiv.org/content/10.64898/2026.11.01.763457v1)

## Authors:
* Yash Shah
* Daniel L.K. Yamins


## Overview of the repository
This repository has the following components:
* `configs/`: YAML files used for model training and model evaluation
* `scripts/`: standalone scripts to be run from the command line.
* `spacetorch/`: installable Python package providing classes and methods for model training, evaluation, plotting, and figure creation
* `analyses/`: code for data analysis

Training and benchmarking scripts can be found under `spacetorch/run/...`

Running files under analyses should take only a few minutes, unless the scripts run images through the models to extract model activations, which can take up to an hour in the absence of a GPU.

In lieu of demos using simulated data, we have provided all configs that can directly be run to train models of various kinds. Simply call `spacetorch/run/train.py` with the appropriate `--config` argument.

## Installation
To install dependencies, follow the instructions in [INSTALL.md](INSTALL.md).

## Citation
Please use the following citation for [the paper](...) or the code in this repository:

>Shah, Y., Yamins, D.L.K. (XXX). XXX. XXX.

_Note: This repository heavily borrows from the [TDANN repository](https://github.com/neuroailab/TDANN). Please refer to that if you have questions around code structure._
