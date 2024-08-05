import pandas as pd
import numpy as np
from pypot.generalized_pareto import param_cov_matrix
from pypot.utils import years_span_series


def compute_r_hat(L, lambda_hat, xi_hat, sigma_hat):
    """Given parameter point estimates, compute point estimate of the
    L year return level.

    args:
        L (int): year for the return level, e.g. 20 for a 20 year return level
        lambda_hat (float): point estimate of lambda from poisson process
        xi_hat (float): point estimate of xi
        sigma_hat (float): point estimate of sigma

    returns:
        (float): point estimate of return level
    """
    first = sigma_hat / xi_hat
    second = (lambda_hat * L)**xi_hat - 1
    return first * second


def compute_lambda_hat(extremes_series):
    """Computes lambda hat as number of exceedences
    divided by number of years, using the time
    series.

    args:
        extremes_series (pd.Series) series of extreme values, with
        a datetime object index

    returns:
        (float) lambda hat MLE
    """
    # make sure extremes series has a datetime type index
    assert isinstance(extremes_series.index, pd.core.indexes.datetimes.DatetimeIndex), \
        "extremes series must have a datetime index"
    # number of exceedences in series
    n = len(extremes_series)

    # time span of the series in years
    years_span = years_span_series(extremes_series)
    # compute lambda hat
    return n / years_span


def normalized_cov_matrix(lambda_hat, xi_hat, sigma_hat, n, t):
    """Computes the normalized covariance matrix
    for lambda, xi, sigma with assumed independence between
    lambda and the other two parameters.

    args:
        theta_hat (length 3 iterable[float]): MLE estimates of lambda, xi, sigma
        n (int): sample size of peaks
        t (float): number of years in the sample period

    returns:
        (np.array[float]): normalized covariance matrix
    """
    # lower 2x2 block of covariance matrix
    # use just xi hat and sigma hat
    # TODO clean this up
    lower_block = param_cov_matrix(xi_hat, sigma_hat, n)
    full_cov_matrix = np.zeros((3, 3))
    full_cov_matrix[1:, 1:] = lower_block
    # normalize lambda hat by dividing by sample size T
    full_cov_matrix[0, 0] = lambda_hat / t

    return full_cov_matrix



def grad_hat_r_l(lambda_hat, xi_hat, sigma_hat, L):
    """Compute gradient of return rate estimator,
    evaluated at theta hat.

    args:
        theta_hat (length 3 iterable[float]): MLE estimates of lambda, xi, sigma
        L (int): desired year return period (i.e. L=20 -> 20 year return period)
    return:
        (np.array[float]): gradient of r_hat_l evaluated at estimators
    """

    # d/dlambda R(L)
    partial_lambda = sigma_hat * L**xi_hat * lambda_hat**(xi_hat - 1)

    # d/dxi R(L)
    partial_xi_1 = sigma_hat / xi_hat * (lambda_hat * L)**xi_hat * np.log(lambda_hat * L)
    partial_xi_2 = ((lambda_hat * L)**xi_hat - 1) * (sigma_hat / xi_hat**2)
    partial_xi = partial_xi_1 - partial_xi_2

    # d/dsigma R(L)
    partial_sigma = ((lambda_hat * L)**xi_hat - 1) / xi_hat
    # full gradient
    gradient = np.array([partial_lambda, partial_xi, partial_sigma])
    gradient.shape = (3, 1)
    return gradient


def var_hat_r_l(lambda_hat, xi_hat, sigma_hat, n, t, L):
    """Approximated variance of return level estimator.

    args:
        {lambda,xi,sigma}_hat (float): MLE estimates of lambda, xi, sigma
        n (int): sample size of peaks
        t (float): number of years in the sample period
        L (int): desired year return period (i.e. L=20 -> 20 year return period)

    returns:
        (float): approximate variance
    """
    Sigma = normalized_cov_matrix(lambda_hat, xi_hat, sigma_hat, n, t)
    grad_R_L = grad_hat_r_l(lambda_hat, xi_hat, sigma_hat, L)
    approx_var = np.dot(np.dot(grad_R_L.transpose(), Sigma), grad_R_L)

    return approx_var
