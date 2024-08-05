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
        numeric: negative log likelihood value
    """
    # SANITY CHECKS
    # make sure that each data point is in the support set,
    # which depends on parameter values
    assert min(x) > 0
    if xi < 0:
        upper_bound = -1 * sigma / xi
        assert max(x) < upper_bound, "observation {0} outside of support constraint {1}".format(
            max(x), upper_bound
        )

    n = len(x)
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
    # TODO bounds
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
    # --> xi max(x_i) + sigma > 0
    max_x = max(x)

    A = [max_x, 1]
    lb = 0  # Lower bound of the constraint 0
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

    print(result.message)
    return result.x