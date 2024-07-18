# PyPoT

This library provides Python routines for fitting Peaks-over-Threshold models of the following forms:

```math
Y_t | \xi, \sigma \sim^{iid} GPD(\xi, \sigma)
```

```math
Y_t | \xi, \beta, X_t \sim^{iid} GPD(\xi, exp(\beta^T X_t))
```

Where GPD is the Generalized Pareto Distribution with the location parameter (threshold) fixed at 0.  The first model is the familiar univariate PoT analysis, and the second allows the $\sigma$ parameter to depend on covariates.
