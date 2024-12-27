import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from functools import partial


def gp_neg_loglik(x, xi, sigma):
    """Negative log likelihood of Generalized Pareto distribution
    in the "xi, sigma" parameterization.

    Args:
        x (np.array): sample values
        xi (numeric): xi parameter
        sigma (numeric): sigma parameter

    Returns:
        numeric: negative log likelihood value, or nan if invalid support
    """
    n = len(x)
    # https://stackoverflow.com/questions/49461299/how-does-scipy-minimize-handle-nans
    # During some optimization iterations (see GPD_fit below) the optimizer will not enforce the linear
    # support constraint, leading to negative numbers in the np.log() here.  This does not mess with the
    # minimizer, which views nans as very large and searches away from them, effectively enforcing
    # the constraint.  We do not want to throw warnings every time, so we ignore the warnings and let
    # the minimizer avoid the support constraint.  If this function is called with x values that violate
    # the support constraint, it will return np.nan.
    with np.errstate(divide='ignore', invalid='ignore'):
        s = sum(np.log(1 + xi * x / sigma))
    ll = -1 * n * np.log(sigma) + (-1 - 1 / xi) * s
    return -1 * ll


def gp_cdf(x, xi, sigma):
    """CDF of the GPD.

    args:
        x (np.array): data
        xi (numeric): xi parameter
        sigma (numeric): sigma parameter

    returns:
        (np.array) values of the cdf at x
    """
    # make sure that each data point is in the support set
    assert min(x) > 0
    if xi < 0:
        upper_bound = -1 * sigma / xi
        assert max(x) < upper_bound, "observations outside of valid support"

    first = (1 + xi * x / sigma)**(-1/xi)
    return 1 - first


def gp_inv_cdf(u, xi, sigma):
    """Inverse CDF of the GPD as paramaterized by the paper above."""
    first = (1 - u)**(-1 * xi) - 1
    second = sigma / xi * first
    return second


def gp_density(x, xi, sigma):
    """Density function for GPD.

    args:
        x (np.array): data
        xi (numeric): xi parameter
        sigma (numeric): sigma parameter

    returns:
        (np.array) values of the density function
    """
    # TODO bounds check option
    first = 1 + xi * x / sigma
    # numpy workaround
    # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
    second = np.sign(first) * (np.abs(first)) ** (-1 - 1/xi)
    third = 1 / sigma * second
    return third


def gp_morans_statistic(x, xi, sigma):
    """Moran's "statistic" for GPD.  The maximum product of spacings
    estimators for GPD parameters minimize this function.

    Args:
        x (np.array): sample values
        xi (numeric): xi parameter
        sigma (numeric): sigma parameter

    Returns:
        numeric: Moran's statistic value
    """
    # sort the sample from low to high
    x = np.sort(x)
    # compute cdf values at sample points
    cdf_vals_samp = gp_cdf(x, xi, sigma)
    # add 0 to beginning and 1 ot end
    prb = np.hstack((0.0, cdf_vals_samp, 1))
    # compute differences between adjacent elements
    dprb = np.diff(prb)
    # compute log differences

    # TODO - set arbitrarily small difference value
    # for datapoints that are far out in the tail of the
    # distribution - the optimization routing will optimize
    # parameter values away from this
    dprb[dprb == 0] = 1e-7

    logD = np.log(dprb)

    # compute Moran's "statistic"
    T = -sum(logD)
    return T


def gp_param_cov_matrix(xi_hat, sigma_hat, n):
    """Compute the covariance matrix for GDP parameter estimates.

    args:
        xi_hat (float): xi estimator value
        sigma_hat (float): sigma estimator value
        n (int): sample size

    returns:
        (2d np.array): covariance matrix
    """
    # unnormalized estimated asymptotic variances
    var_xi_hat = (1 - xi_hat**2)
    var_sigma_hat = 2 * sigma_hat **2 * (1 + xi_hat)
    # unnormalized estimated asymptotic covariance
    cov_xi_sigma_hat = -1 * sigma_hat * (1 + xi_hat)
    # unnormalized cov matrix
    cov_matrix = np.array([
        [var_xi_hat, cov_xi_sigma_hat],
        [cov_xi_sigma_hat, var_sigma_hat]
    ])
    # normalize
    return n**(-1) * cov_matrix



def gp_neg_loglik_jacob(theta, x):
    """Jacobian of the GPD negative log likelihood, used
    in the sequential quadratic programming optimization routine.

    args:
        theta (array[float]): (xi, sigma) params
        x (array[float]): data
    returns:
        (array[float]) value of the jacobian
    """
    xi = theta[0]
    sigma = theta[1]
    n = len(x)

    # partial derivatives
    d_dxi = (1 + 1 / xi) * np.sum(x / (sigma + xi * x)) - np.sum(np.log(1 + xi * x / sigma)) * xi ** (-1 * 2)
    d_dsigma = n / sigma - (1 + 1 / xi) * np.sum(x * xi / (sigma ** 2 + sigma * x * xi))

    jacob = np.array([d_dxi, d_dsigma])
    return jacob


def gp_neg_loglik_hess(theta, x):
    """Hessian of the GPD negative log likelihood, for use in
    optimization procedures that require it.

    args:
        theta (array[float]): (xi, sigma) params
        x (array[float]): data
    returns:
        (2x2 array[float]) value of the hessian
    """
    xi = theta[0]
    sigma = theta[1]
    n = len(x)

    # second order partials
    # ==================================================
    # d^2 / dxi^2
    a1 = np.sum(np.log(1 + xi * x / sigma))
    a2 = np.sum(x / (sigma + xi * x))
    a3 = np.sum(x**2 / (sigma + xi * x)**2)
    d_dxi_sq = (2 / xi**3) * a1 - (2 / xi**2) * a2 - (1 + 1 / xi) * a3

    # d^2 / dsigma^2
    sum_term = np.sum(xi * x * (2 * sigma + xi * x) / (sigma ** 2 + sigma * xi * x)^2)
    d_dsigma_sq = (1 + 1 / xi) * sum_term - n / sigma**2

    # d^2 / dxi dsigma
    s1 = np.sum(xi * x / (sigma**2 + sigma * x * xi))
    s2 = np.sum(x / (sigma + x * xi)**2)
    d_dsigma_dxi = 1 / xi**2 * s1 - (1 + 1 / xi) * s2

    hessian = np.array([[d_dxi_sq, d_dsigma_dxi], [d_dsigma_dxi, d_dsigma_sq]])
    return hessian


def fit_GPD(x, theta_0, f_minimize, jacobian=None):
    """Parameter estimation by objective function minimization,
    via scipy optimizer.

    x: 1d np array, data values
    theta_0: 2-tuple[float], initial guess of values
    f_minimize: callable function with call signature:
        f_minimize(x, xi, sigma) = r
    jacobian: callable function with call signature:
        jacobian([xi, sigma], x), the jacobian of the objective function
        to minimize

    return: np.array[float] point estimates
    """
    # partially apply negative log likelihood at given X values
    f_partial = partial(f_minimize, x=x)

    # routine to optimize
    def minimize_me(theta):
        xi, sigma = theta
        return f_partial(xi=xi, sigma=sigma)

    # bounds and constraint
    # =========================================
    # sigma > 0
    # xi in a range that produces finite support
    # TODO specify this bound on xi as an option
    bounds = [(-1 / 2, 1 / 2), (0.01, None)]

    # constraint on xi
    # xi > -sigma / max(x)
    # --> xi + sigma / max(x) > 0
    max_x = max(x)

    # scaling factor to avoid numerical precision issues
    scale_factor = 1000
    A = [scale_factor, scale_factor / max_x]
    # constraint bounds - "true" lower bound will be lb / scale_factor
    lb = 0.01
    ub = float('inf')

    lin_constraint = LinearConstraint(A, lb, ub)

    # check if jacobian is provided
    if jacobian is not None:
        jacob = partial(jacobian, x=x)
    else:
        jacob = None

    # =========================================

    result = minimize(
        minimize_me,
        theta_0,
        bounds=bounds,
        constraints=[lin_constraint],
        method="SLSQP",
        jac=jacob,
    )
    # ensure optimization routine converged
    if result.message != "Optimization terminated successfully":
        raise RuntimeError("optimization failed in fit_GPD: {0}".format(result.message))

    # sanity checks for support - make sure the optimizer respected the constraint
    # check this after the fact as the optimizer will not enforce the constraint at every iteration
    # https://stackoverflow.com/questions/47682698/scipy-optimization-with-slsqp-disregards-constraints
    xi_hat = result.x[0]
    sigma_hat = result.x[1]
    if xi_hat < 0:
        upper_bound = -1 * sigma_hat / xi_hat
        assert max_x < upper_bound, "observation {0} outside of support constraint {1}, xi_hat={2}, sigma_hat={3}".format(
            max_x, upper_bound, xi_hat, sigma_hat
        )

    return result.x


def anderson_darling_statistic(x, xi, sigma):
    """Anderson Darling statistic for GPD.

    args:
        x (np.array): data
        xi (float): xi parameter
        sigma (float): sigma parameter

    returns:
        (float) value of the statistic
    """
    n = len(x)
    # sort the sample from low to high
    x = np.sort(x)
    # compute cdf values at sample points
    cdf_vals_samp = gp_cdf(x, xi, sigma)
    # reverse the cdf vals
    # i.e. (z(1), z(2), ... z(n)) -> (z(n), z(n-1), ... z(1))
    cdf_vals_desc = cdf_vals_samp[::-1]
    # array of 2i - 1 for i=1, ..., n
    coeff = 2 * np.arange(1, n + 1) - 1
    # sum term
    s = np.sum(coeff * (np.log(cdf_vals_samp) + np.log(1 - cdf_vals_desc)))
    return -1 * n - s/n
