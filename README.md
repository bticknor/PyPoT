# PyPoT

## Features

This library provides Python routines for fitting the Peaks-over-Threshold statistical models of the form:

```math
Y_t | x_t \sim^{ind.} GPD(\xi, \sigma_t)
```

Where:

```math
\sigma_t = exp(x_t^T \beta)
```

For p-dimensional parameter vector $\beta$, where GPD corresponds to the Generalized Pareto Distribution with the location parameter fixed at 0.  The following functionality is available:

- Maximum likelihood point estimation of model parameters, and asymptotic uncertainty quantification
- Return level estimation and asymptotic uncertainty quantification
- Statistically principled automatic threshold selection using the ForwardStop algorithm utilizing Anderson-Darling hypothesis tests

This is a statistical library with a very limited scope. Routines for visualizing extreme values can be found in (e.g.) [PyExtremes](https://github.com/georgebv/pyextremes).

## Installation

Clone the repository, navigate to the root directory, and run:

`pip install .`

## Documentation

Mathematical documentation that includes definitions and expressions for asymptotic results can be found [here](stats.md).  A Jupyter notebook with examples code for performing PoT analysis tasks can be found [here](pypot/examples/PyPoT_examples.ipynb).

## TODOs

- Update examples notebook
- Implement uncertainty quantification for beta parameters