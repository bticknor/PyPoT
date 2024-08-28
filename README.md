# PyPoT

## Features

This library provides Python routines for fitting the univariate Peaks-over-Threshold analysis:

```math
Y_t | \xi, \sigma \sim^{iid} GPD(\xi, \sigma)
```

Where GPD is the Generalized Pareto Distribution with the location parameter fixed at 0.  The following functionality is available:

- Maximum likelihood and maximum product of spacings point estimation of model parameters, and asymptotic uncertainty quantification
- Return level estimation and asymptotic uncertainty quantification
- Statistically principled automatic threshold selection using the ForwardStop algorithm via Anderson-Darling hypothesis tests, as described by Bader et. all (2018)

This is a statistical library with a very limited scope. Routines for visualizing extreme values can be found in (e.g.) [PyExtremes](https://github.com/georgebv/pyextremes).

## Installation

To install PyPoT via pip, clone the repository to your machine, navigate to the root directory, and run:

`pip install .`

## Documentation

Mathematical "documentation" that includes definitions and expressions for asymptotic results can be found [here](stats.md).  A Jupyter notebook with examples code for performing PoT analysis tasks can be found [here](pypot/examples/PyPoT_examples.ipynb).  Full documentation of all available routines is still under construction.

## TODOs

- Documentation
- Implement gradient for Moran's statistic for faster fitting of MPS estimators
