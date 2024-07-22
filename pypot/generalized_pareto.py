import numpy as np


def negative_log_likelihood(x, xi, sigma):
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
        assert max(x) < upper_bound, "observations outside of valid support"

    n = len(x)
    s = sum(np.log(1 + xi * x / sigma))
    ll = -1 * n * np.log(sigma) + (-1 - 1 / xi) * s
    return -1 * ll


def cdf(x, xi, sigma):
    """CDF of the GPD as parameterized by the paper linked above.

    x is a numpy array
    """
    # SANITY CHECKS
    # make sure that each data point is in the support set
    assert min(x) > 0
    if xi < 0:
        upper_bound = -1 * sigma / xi
        assert max(x) < upper_bound, "observations outside of valid support"

    first = (1 + xi * x / sigma)**(-1/xi)
    return 1 - first


def morans_statistic(x, xi, sigma):
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
    cdf_vals_samp = gpd_cdf(x, xi, sigma)
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


def param_cov_matrix(theta_hat, n):
    """Computes the asymptotically approximated covariance matrix
    for efficient estimators (e.g. MLE or MPS) of GPD xi and sigma
    params.

    Args:
        theta_hat (iterable): point estimates
        n (int): sample size

    Returns:
        np.ndarray: covariance matrix
    """
    # unpack MLE point estimates
    xi_hat = theta_hat[0]
    sigma_hat = theta_hat[1]
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
