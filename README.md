# PyPoT

## Features

This library provides Python routines for fitting Peaks-over-Threshold models of the following forms:

```math
Y_t | \xi, \sigma \sim^{iid} GPD(\xi, \sigma)
```

```math
Y_t | \xi, \beta, X_t \sim^{iid} GPD(\xi, exp(\beta^T X_t))
```

Where GPD is the Generalized Pareto Distribution with the location parameter fixed at 0.  The first model is the familiar univariate PoT analysis, and the second allows the $\sigma$ parameter to depend on covariates.  The following functionality is available:

- Maximum likelihood and maximum product of spacings point estimation of model parameters, and asymptotic uncertainty quantification
- Return rate estimation and asymptotic uncertainty quantification

This is a statistical library with a very limited scope. Routines for extracting and plotting extreme values can be found in (e.g.) [PyExtremes](https://github.com/georgebv/pyextremes).

## Documentation

Detailed mathematical "documentation" that includes expressions for asymptotic results can be found [here](stats.md).  Example usage of the software can be found TODO!

## Usage

TODO!

## TODOs

- Implement gradient for Moran's statistic for faster fitting of MPS estimators