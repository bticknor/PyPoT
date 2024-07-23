# PyPoT Statistical Details 

## Generalized Pareto Distribution

I use the $\xi, \sigma$ parameterization of the Generalized Pareto distribution, which yields the following CDF and PDF:

```math
    F_{\xi, \sigma}(x) = 1 - (1 + \xi \frac{x}{\sigma})^{-1/\xi}
```

```math
    f_{\xi, \sigma}(x) = (\frac{1}{\sigma}) (1 + \xi \frac{x}{\sigma})^{-1 - \xi^{-1}}
```

The location parameter is fixed at 0, as the threshold is subtracted from peaks before inference.

## Maximum Likelihood Estimation

```math
    \displaylines{a = b \\
     c = d}
```

## Maximum Product of Spacings Estimation

TODO mention asymptotics are the same


## Approximate Covariance Matrix for MLE/MPS estimates



## Return Level Estimates




## Approximate Variance for Return Level Estimates





TODO!