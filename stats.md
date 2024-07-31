# PyPoT Statistical Details 

## Generalized Pareto Distribution

I use the $\xi, \sigma$ parameterization of the Generalized Pareto distribution, which yields the following CDF and PDF:

```math
    F_{\xi, \sigma}(x) = 1 - (1 + \xi \frac{x}{\sigma})^{-1/\xi}
```

```math
    f_{\xi, \sigma}(x) = (\frac{1}{\sigma}) (1 + \xi \frac{x}{\sigma})^{-1 - \xi^{-1}}
```

Where $x \in [0, \infty)$ for $\xi \geq 0$ and $x \in [0, -\frac{\sigma}{\xi}]$ for $\xi < 0$.  The location parameter is fixed at 0, as the threshold is assumed known and subtracted from the peaks observations before inference.

## Maximum Likelihood Estimators

The maximum likelihood estimators minimize the GPD negative log likelihood:

```math
    \displaylines{ (\hat{\xi}, \hat{\sigma}) = argmin_{\xi, \sigma}  -l(\xi, \sigma | X) \\
    -l(\xi, \sigma | X) = -nlog(\sigma) - (1 + \frac{1}{\xi}) \sum_{i=1}^n log(1 + \xi \cdot \frac{x_i}{\sigma}) }
```

## Maximum Product of Spacings Estimators

The maximum product of spacings estimators minimize the "Moran's statistic":

```math
    \displaylines{ (\tilde{\xi}, \tilde{\sigma}) = argmin_{\xi, \sigma}  M(\xi, \sigma | x) \\ 
    M(\xi, \sigma | x) = - \sum_{i=1}^{n+1} log \left(  F_{\xi, \sigma}(x_{(i)}) - F_{\xi, \sigma}( x_{(i-1)}  \right) }
```

Where $x_{(i)}$ is the $i$'th element in the ordered sample, with $x_0 \equiv 0$, and $x_{(n+1)} \equiv 1$.  MPS estimators have the same asymptotic properties as the MLEs (when the MLEs exist), e.g. asymptotic efficiency (TODO CITE https://arxiv.org/pdf/math/0702830
).

## Poisson Process Parameter Estimator

TODO!

```math
    \hat{\lambda} = ...
```

## Return Level Estimator

An estimator for the $L$ year return level is:

```math
    \widehat{R(L | \hat{\theta})} = \frac{\hat{\sigma}}{\hat{\xi}} \left((\hat{\lambda} L)^{\hat{\xi}} - 1   \right)
```

This is derived from computing the expected time between $L$ year exceedences, given GPD parameter estimates. TODO CITE OUR PAPER


## Approximate Variance for MLE/MPS Estimators and Return Level Estimator

Letting $\theta = (\lambda, \sigma, \xi)$ and assuming independence between $\lambda$ and the GPD parameters, we have the following asymptotic covariance matrix:

```math
    cov(\hat{\theta}) \equiv \Sigma = \begin{bmatrix}
        \frac{\lambda}{T} & 0 & 0 \\
        0 & \frac{2 \sigma^2 (1 + \xi)}{N(T)} & \frac{-\sigma(1 + \xi)}{N(T)} \\
        0 & \frac{-\sigma(1 + \xi)}{N(T)} & \frac{(1 - \xi^2)}{N(T)}
    \end{bmatrix}
```

Where $T$ is the length of the time series in years and $N(T)$ is the number of exceedences given $T$.  Letting $h(\theta) = \widehat{R(L | \theta)}$, we have by the delta method that the asymptotic variance of the return level estimator is:

```math
    Var(\widehat{R(L)}) \approx \nabla h(\theta)^T \cdot  \Sigma \cdot \nabla h(\theta)
```

With:

```math
    \nabla h(\theta) = \begin{bmatrix}
        \sigma L^\xi \lambda^{\xi - 1} \\
        \frac{\sigma}{\xi} (\lambda L)^\xi log(\lambda L) - \left( (\lambda L)^\xi - 1 \right) (\frac{\sigma}{\xi^2}) \\
        \frac{(\lambda L)^\xi - 1}{\xi}
    \end{bmatrix}
```

The estimate of this is computed as:

```math
    \widehat{Var}(\widehat{R(L)}) = \nabla h(\theta)^T \cdot  \Sigma \cdot \nabla h(\theta)|_{\theta = \hat{\theta}}
```

TODO CITE

https://www.stat.cmu.edu/technometrics/80-89/VOL-29-03/v2903339.pdf


## Threshold selection

TODO!

