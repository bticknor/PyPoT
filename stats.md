# PyPoT Statistical Details 

## Generalized Pareto Distribution

The $\xi, \sigma$ parameterization of the Generalized Pareto distribution is used, which yields the following CDF and PDF:

```math
    F_{\xi, \sigma}(y) = 1 - (1 + \xi \frac{y}{\sigma})^{-1/\xi}
```

```math
    f_{\xi, \sigma}(y) = (\frac{1}{\sigma}) (1 + \xi \frac{y}{\sigma})^{-1 - \xi^{-1}}
```

Where $y \in [0, \infty)$ for $\xi \geq 0$ and $y \in [0, -\frac{\sigma}{\xi}]$ for $\xi < 0$.  The location parameter is fixed at 0, as the threshold is assumed known and subtracted from the peaks observations before inference.

## Maximum Likelihood Estimators

For the univariate model, the maximum likelihood estimators minimize the GPD negative log likelihood:

```math
    \displaylines{ (\hat{\xi}, \hat{\sigma}) = argmin_{\xi, \sigma}  -\ell(\xi, \sigma | y) \\
    -\ell(\xi, \sigma | y) = nlog(\sigma) + (1 + \frac{1}{\xi}) \sum_{i=1}^n log(1 + \xi \cdot \frac{y_i}{\sigma}) }
```

The jacobian of which is determined by:

```math
    \frac{d}{d\xi} [-\ell(\xi, \sigma | y)] = (1 + \frac{1}{\xi}) \left(\sum_{i=1}^n \frac{y_i}{\sigma + \xi y_i}  \right) - \left( \sum_{i=1}^n log(1 + \xi \cdot \frac{y_i}{\sigma}) \right) \cdot \frac{1}{\xi^2} 
```

```math
    \frac{d}{d\sigma} [-\ell(\xi, \sigma | y)] = \frac{n}{\sigma} - (1 + \frac{1}{\xi}) \left( \sum_{i=1}^n \frac{y_i \xi}{\sigma^2 + \sigma y_i \xi} \right)
```

For the extended model with $\sigma_t = exp(x_t^T \beta)$, the element of the Jacobian corresponding to $\beta_j$ is:

```math
    \frac{d}{d\beta_j} [-\ell(\xi, \beta | y,x)] = \sum_{t=1}^n x_{t,j}\left( 1 - (1 + \frac{1}{\xi}) \frac{\xi y_t}{\sigma_t + \xi y_t} \right) \\
    \implies \frac{d^2}{d \beta_j^2} [-\ell(\xi, \beta | y,x)] = (1 - \frac{1}{\xi}) \sum_{t=1}^n \frac{\xi y_t}{(\sigma_t + \xi y_t)^2} \sigma_t x_{t,j}
```


## Maximum Product of Spacings Estimators

The maximum product of spacings estimators minimize the "Moran's statistic":

```math
    \displaylines{ (\tilde{\xi}, \tilde{\sigma}) = argmin_{\xi, \sigma}  M(\xi, \sigma | y) \\ 
    M(\xi, \sigma | y) = - \sum_{i=1}^{n+1} log \left(  F_{\xi, \sigma}(y_{(i)}) - F_{\xi, \sigma}( y_{(i-1)}  \right) }
```

Where $y_{(i)}$ is the $i$'th element in the ordered sample, with $y_0 \equiv 0$, and $y_{(n+1)} \equiv 1$.  MPS estimators have the same asymptotic properties as the MLEs (when the MLEs exist), e.g. asymptotic efficiency [1].

## Poisson Process Parameter Estimator

The MLE for the parameter governing the Poisson process that generates exceedences in the PoT model is given by:

```math
    \hat{\lambda} = \frac{N(T)}{T}
```

Where $T$ is the time span of the series in years and $N(T)$ is the number of independent exceedences.

## Return Level Estimator

An estimator for the $L$ year return level is:

```math
    \widehat{R(L | \hat{\theta})} = \frac{\hat{\sigma}}{\hat{\xi}} \left((\hat{\lambda} L)^{\hat{\xi}} - 1   \right)
```

This is derived from computing the expected time between $L$ year exceedences, given GPD parameter estimates.


## Approximate Variance for MLE/MPS Estimators and Return Level Estimator

Letting $\theta = (\lambda, \sigma, \xi)$ and assuming independence between $\lambda$ and the GPD parameters, we have the following asymptotic covariance matrix:

```math
    cov(\hat{\theta}) \equiv \Sigma = \begin{bmatrix}
        \frac{\lambda}{T} & 0 & 0 \\
        0 & \frac{2 \sigma^2 (1 + \xi)}{N(T)} & \frac{-\sigma(1 + \xi)}{N(T)} \\
        0 & \frac{-\sigma(1 + \xi)}{N(T)} & \frac{(1 - \xi^2)}{N(T)}
    \end{bmatrix}
```

Letting $h(\theta) = \widehat{R(L | \theta)}$, we have by the delta method that the asymptotic variance of the return level estimator is:

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

For more see [2].

## Threshold Selection

PyPoT includes an implementation of the goodness-of-fit based threshold selection procedure given by Bader et. all [3].  PyPoT uses the ForwardStop algorithm with the Anderson-Darling hypothesis test, where the AD statistic is defined as:

```math
    A_n^2 = -n - \frac{1}{n} \sum_{i=1}^n (2i - 1) [log(z_{(i)}) + log(1 - z_{(n + 1 - i)}) ]
```

Where:

```math
    z_{(i)} = F_{\hat{\xi}, \hat{\sigma}}(y_{(i)})
```

P-values for this test are computed using the critical value table [here](pypot/data/ADQuantiles.csv).  This critical value table and code for interpolating between values is translated directly from [here](https://github.com/brianbader/eva_package/tree/master).  The ForwardStop algorithm chooses from $l$ ordered hypothesis tests using the logic:

```math
    \hat{k}_F = max \Biggl\{ k \in \{1, ..., l\}: \; -\frac{1}{k} \sum_{i=1}^k log(1 - p_i) \leq \alpha \Biggr\}
```

For false discovery rate control $\alpha$.  See [3] for more details.


## References

[1] Wong, T. S. T., & Li, W. K. (2006). A Note on the Estimation of Extreme Value Distributions Using Maximum Product of Spacings. Lecture Notes-Monograph Series, 52, 272-283.

[2] Hosking, J.R. and Wallis, J.R. (1987) Parameter and Quantile Estimation for the Generalized Pareto Distribution. Technometrics, 29, 339-349.

[3] Bader, Brian & Yan, Jun & Zhang, Xuebin. (2018). Automated threshold selection for extreme value analysis via ordered goodness-of-fit tests with adjustment for false discovery rate. The Annals of Applied Statistics. 12. 310-329. 10.1214/17-AOAS1092.
