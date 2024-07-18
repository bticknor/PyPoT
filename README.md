# PyPoT
Peaks over threshold modelling in Python

This library provides routines for fitting PoT models of the following forms:

```math
Y_t | \xi, \sigma \sim^{iid} GPD(\xi, \sigma)
```

```math
Y_t | \xi, \beta, X_t \sim^{iid} GPD(\xi, exp(\beta^T X_t)
```

The first model is the familiar univariate PoT analysis, and the second allows the $\sigma$ parameter to depend on covariates.
