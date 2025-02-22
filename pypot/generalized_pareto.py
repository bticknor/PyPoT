import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from functools import partial


def gp_nll(params, y, X):
    """
    Negative log-likelihood for the GPD model, optionally with covariates

    params: array-like, parameters to optimize [xi, beta_0, beta_1, ..., beta_p]
    y: array-like, observed data (response variable)
    X: array-like, design matrix for covariates (n_samples, p_features)
    """
    xi = params[0]  # Shape parameter

    p = X.shape[1]
    beta = params[1:].reshape((p, 1))  # Coefficients for the linear predictor

    # TODO parameterize link function
    sigma = np.exp(X @ beta)  # Scale parameter (positive by construction)

    # Check the validity of the support
    if np.any(1 + xi * y / sigma <= 0):  # Ensure the support is satisfied
        return np.inf

    # Compute the negative log-likelihood
    term1 = np.sum(np.log(sigma))  # Scale term
    term2 = np.sum((1 + 1 / xi) * np.log(1 + xi * y / sigma))  # Shape-dependent term

    return term1 + term2


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
        if np.any(x > upper_bound):
            raise RuntimeError("observations outside of valid support")

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


def gp_neg_loglik_jacob(theta, y):
    """Jacobian of the univariate GPD negative log likelihood, can be
    used in gradient based optimization methods.

    args:
        theta (array[float]): (xi, beta) params
        x (array[float]): data
    returns:
        (array[float]) value of the jacobian
    """
    xi = theta[0]
    beta = theta[1]
    sigma = np.exp(beta)
    n = len(y)

    # partial derivatives
    d_dxi = (1 + 1 / xi) * np.sum(y / (sigma + xi * y)) - np.sum(np.log(1 + xi * y / sigma)) * xi ** (-1 * 2)
    d_dsigma = n / sigma - (1 + 1 / xi) * np.sum(y * xi / (sigma ** 2 + sigma * y * xi))
    # chain rule
    d_dbeta = d_dsigma * np.exp(beta)

    jacob = np.array([d_dxi, d_dbeta])
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


def fit_GPD(data, y_lab, x_lab, method="SLSQP"):
    """Parameter estimation via negative log likelihood
    minimization.  Choice of gradient based or non gradient based
    methods.

    data: pd.DataFrame of observations that includes outcome variable and covs
    y_lab: str column name of the outcome
    x_lab: list[string] or None column names of covariates

    returns: np.array([xi_hat, beta_1_hat, ..., beta_p+1_hat])
    """
    assert method in ["SLSQP", "diff_evo"], "method must be either `SLSQP` or `diff_evo`"

    # covariate data
    data["intercept"] = 1
    X_cols = ["intercept"] + x_lab
    covariates = data[X_cols]

    # design matrix
    X = covariates.to_numpy()
    p = X.shape[1]

    # response vector
    y = data[y_lab].to_numpy()
    y = y.reshape(len(y), 1)

    # TODO parameterize these and update lower bound
    bounds = [(-1/2, 1/2)] + [(-5, 5) for _ in range(p)]

    if method == "SLSQP":
        # fit model using sequential least squares

        # TODO! implement jacobian for p>1
        if p > 1:
            jac = None

        else:
            def jac(params, y, X):
                """Necessary for using bounds kwarg in minimize."""
                return gp_neg_loglik_jacob(params, y)

        # initialize betas to 0
        theta_init = np.array([1/5] + [np.log(1) for _ in range(p)])
        result = minimize(
            gp_nll,
            x0=theta_init,
            bounds=bounds,
            args = (y, X),
            method=method,
            jac=jac
        )

    elif method == "diff_evo":
        # fit model using differential evolution
        result = differential_evolution(
            gp_nll,
            bounds,
            args = (y, X),
            strategy="best1bin",
            maxiter=1000,
            tol=1e-4,
            disp=False
        )

    # # ensure optimization routine converged
    # come on now
    success_message = {
        "SLSQP": "Optimization terminated successfully",
        "diff_evo": "Optimization terminated successfully."
    }

    if result.message != success_message[method]:
        raise RuntimeError("likelihood optimization failed: {0}".format(result.message))

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


def beta_obs_fisher_info(y, X, theta_hat):
    """Observed fisher information for beta parameters.

    args:
        X (np.array): matrix of covariate values
        y (np.array): observation vector
        theta_hat (np.array): MLE (xi_hat, beta_1_hat, ..., beta_p+1_hat)
    
    returns:
        np.array([float]) array of observed Fisher information values
    """
    xi_hat = theta_hat[0]
    beta_hat = theta_hat[1:]
    # make this a column vector
    beta_hat = beta_hat.reshape(1, -1).T

    # vector of sigma hats
    sigma_hat_t = np.exp(X @ beta_hat)

    # reshape y into column vector
    y = y.reshape(1, -1).T

    c = 1 + 1 / xi_hat

    # element wise multiplications here
    num = xi_hat * y * sigma_hat_t
    denom = (sigma_hat_t + xi_hat * y)**2
    C_t = num / denom

    X_sq = X ** 2
    # observed Fisher information
    J = c * X_sq.T @ C_t

    # reshape back into 1d array
    return J.reshape(-1)
