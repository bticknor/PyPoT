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

Letting $\theta = (\xi, \sigma)$, the maximum likelihood estimators minimize the negative log likelihood:

TODO!

```math
    \displaylines{a = b \\
     c = d}
```

## Maximum Product of Spacings Estimators

The maximum product of spacings estimators minimize the "Moran's statistic":

```math
    \displaylines{ (\tilde{\xi}, \tilde{\sigma}) = argmin_{\xi, \sigma}  M(\xi, \sigma | x) \\ 
    M(\xi, \sigma | x) = - \sum_{i=1}^{n+1} log \left(  F_{\xi, \sigma}(x_{(i)}) - F_{\xi, \sigma}( x_{(i-1)}  \right) }
```

Where $x_{(i)}$ is the $i$'th element in the ordered sample, with $x_0 \equiv 0$, and $x_{(n+1)} \equiv 1$.  MPS estimators have the same asymptotic properties as the MLEs (when the MLEs exist), e.g. asymptotic efficiency (TODO CITE https://arxiv.org/pdf/math/0702830
).


## Asymptotic Covariance Matrix for MLE/MPS Estimators

Asymptotically, efficient estimators of $\xi$ and $\sigma$ have the following covariance matrix:

```math
     \displaylines{ n \cdot var(\begin{bmatrix}
        \hat{\sigma}\\
        \hat{\xi}
        \end{bmatrix}
    ) \approx \begin{bmatrix}
        2 \sigma^2 (1 + \xi) & -\sigma(1 + \xi) \\
        -\sigma(1+\xi) & (1 - \xi^2)
    \end{bmatrix} }
```
This is estimated by setting $\theta$ to either the MLE or MPS estimates.

TODO CITE

https://www.stat.cmu.edu/technometrics/80-89/VOL-29-03/v2903339.pdf


## Poisson Process Estimator

TODO!

## Return Level Estimator

An estimator for the $L$ year return level is:

```math
    \hat{R}(L) = \frac{\hat{\sigma}}{\hat{\xi}} \left((\hat{\lambda} L)^{\hat{\xi}} - 1   \right)
```

This is derived from computing the expected time between $L$ year exceedences, given GPD parameter estimates. TODO CITE OUR PAPER

## Approximate Variance for Return Level Estimator

Assuming independence between $\lambda$ and the GPD parameters, we have that:

TODO



TODO!